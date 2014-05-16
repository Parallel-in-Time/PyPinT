# coding=utf-8
import numpy as np
import sys
print(sys.path)
from pypint.plugins.multigrid.multigrid_problem import MultiGridProblem
from pypint.plugins.multigrid.multigrid_level_provider import MultiGridLevelProvider
from pypint.plugins.multigrid.multigrid_solution import MultiGridSolution
from pypint.plugins.multigrid.level import MultigridLevel1D
from pypint.plugins.multigrid.level2d import MultigridLevel2D
from pypint.plugins.multigrid.multigrid_smoother import SplitSmoother,ILUSmoother, DirectSolverSmoother, WeightedJacobiSmoother
from pypint.utilities import assert_is_callable, assert_is_instance, assert_condition
from pypint.plugins.multigrid.stencil import Stencil
from pypint.plugins.multigrid.interpolation import InterpolationByStencilListIn1D, InterpolationByStencilForLevels, InterpolationByStencilForLevelsClassical
from pypint.plugins.multigrid.restriction import RestrictionStencilPure, RestrictionByStencilForLevels, RestrictionByStencilForLevelsClassical
from operator import iadd,add
from pypint.plugins.multigrid import MG_INTERPOLATION_PRESETS, MG_RESTRICTION_PRESETS, MG_SMOOTHER_PRESETS, MG_LEVEL_PRESETS
import matplotlib.pyplot as plt

class MultiGridControl(object):
    """
    Summary
    _______
    Contains some functions which, for example check if the multigrid is done,
    it also generates the control flow iterator/string(not sure about it yet).
    gathers the matrices.(Weiss unter anderem ob eine bestimmte matrix fuer ein
    bestimmtes level schon existiert)
    """
    def __init__(self, *args, **kwargs):

        self.where_i_am = 0
        self.where_to_go = 1
        self.where_i_were = 0
        self.ascending = False
        # MultigridControl should be able to work with the following
        # three possibilities in which
        self.lvl_dict = kwargs.get("level_dict", None)
        self.lvl_graph = kwargs.get("level_graph", None)
        self.lvl_list = kwargs.get("level_list", None)
        # one has to now what happened on each level so far.
        self.history = []

    def i_am_on_top(self):
        """Returns True if one is on the top

        """
        if self.where_i_am == 0:
            return True
        else:
            return False

    def i_am_deep_down(self):
        """Returns True if one is at one end

        """
        if self.where_i_am == (len(self.lvl_list) - 1):
            return True
        else:
            return False

    def decide_where_to_go(self):
        """sets self.where_to_go and ascending variable

        this method should be overwritten by a subclass in order to
        change the control flow , e.g. from a v cycle to a w cycle
        or something completely different. In this example we class
        we implement a simple endless V cycle
        """
        if self.i_am_deep_down():
            self.where_to_go = self.where_i_am - 1
            self.ascending = True
        elif self.i_am_on_top():
            self.where_to_go = self.where_i_am + 1
            self.ascending = False
        else:
            if self.ascending:
                self.where_to_go = self.where_i_am - 1
            else:
                self.where_to_go = self.where_i_am + 1
        # Log the information
        self.history.append("Decided to go to level " +
                            self.lvl_list[self.where_to_go])

    def go_to_next_level(self):
        """Organizes the move to the next step

        returns a list of command dicts which are given
        to the MultiGridLevelProvider do method
        """
        if self.where_i_am < self.where_to_go:
            command = {"from_level": self.lvl_list[self.where_i_am],
                       "to_level": self.lvl_list[self.where_to_go],
                       "restriction": ""}

        else:
            command = {"from_level": self.lvl_list[self.where_i_am],
                       "to_level": self.lvl_list[self.where_to_go],
                       "interpolation": ""}

        self.where_i_were = self.where_i_am
        self.where_i_am = self.where_to_go
        self.decide_where_to_go()
        self.history.append("Went from level " + str(self.where_i_were) +
                            " to level " + str(self.where_i_am))
        return command

    def go_to_level(self, level):
        pass

    def relax(self, n_times, smoother=""):
        """Gives the command to relax n_times

        """
        command = {"level": self.lvl_list[self.where_i_am],
                   "smoother": smoother,
                   "smooth_n_times": n_times}
        self.history.append("Applied the smoother "
                            + smoother + " " + str(n_times))
        return command


class ResidualErrorControl(MultiGridControl):
    """Controls the Flow by checking the residual error, has methods to measure
       the residual

    """
    def __init__(self, residual_tolerance_dict, max_iteration_dict,
                 *args, **kwargs):
        self.rtd = residual_tolerance_dict
        self.mid = max_iteration_dict
        if "res_comp_method" in kwargs.keys():
            self.compute_residual = kwargs["res_comp_method"]
            assert_is_callable(self.compute_residual,
                               "Residual computation method "
                               "should be callable ")
        else:
            self.compute_residual = self._standard_residual_computation
        super(ResidualErrorControl, self).__init__(args, kwargs)

    def _standard_residual_computation(self, prev_arr, act_arr):
        """standard residual computation

        """
        return np.sum(np.abs(prev_arr - act_arr))



class MultiGridCore(object):
    """
    Parameters
    ----------
    MultiGridProblem mgprob
        Contains the description of the problem in the multigrid language.
    MultiGridLevelProvider mglprov
        Contains the management of the different grids and the interpolation
        and restriction operators.
    MultiGridSolution
        Like a filter which takes every output of the multigrid run
        and saves it in useful format.
    Summary
    -------
    The main ingredients of MultiGrid are merged in this class,
    the most important function is the run command
    """

    def __init__(self, mg_prob, stencil_form, *args, **kwargs):

        for keys in kwargs.keys():
            print(keys)
        assert_is_instance(mg_prob, MultiGridProblem, "Not a multigrid Problem")
        assert_is_callable(stencil_form, "StencilForm has to be a function")
        self.mg_problem = mg_prob
        self.levels = []
        self.smoothers = []
        self.stencils = []
        self.rst_ops = []
        self.ipl_ops = []
        self.dim = kwargs["dim"]
        self.n_pre = kwargs.get("n_pre", 3)
        self.n_post = kwargs.get("n_post", 3)
        self.num_levels = kwargs.get("num_levels", 3)

        # append course level
        shape = kwargs["shape_coarse"]
        if kwargs.get("dim") == 1:
            self.levels.append(MultigridLevel1D(shape, self.mg_problem,
                                                max_borders=kwargs["max_borders"], role="CL"))
        elif kwargs.get("dim") == 2:
            self.levels.append(MultigridLevel2D(shape, self.mg_problem,
                                                max_borders=kwargs["max_borders"], role="CL"))
        #append course stencil
        self.stencils.append(Stencil(*stencil_form(self.levels[-1])))
        self.smoothers.append(DirectSolverSmoother(self.stencils[-1], self.levels[-1]))


        for i in range(kwargs["num_levels"]-1):

            if i == kwargs["num_levels"]-2:
                role = "FL"
            else:
                role = "ML"
            if kwargs.get("dim") == 1:
                shape = shape*2+1
                self.levels.append(MultigridLevel1D(shape, self.mg_problem,
                                                    max_borders=kwargs["max_borders"], role=role))
            elif kwargs.get("dim") == 2:
                shape = (shape[0]*2+1, shape[1]*2+1)
                self.levels.append(MultigridLevel2D(shape, self.mg_problem,
                                                    max_borders=kwargs["max_borders"], role=role))

            self.stencils.append(Stencil(*stencil_form(self.levels[-1])))
            # append smoother
            if kwargs["smoothing_type"] is "jacobi":
                omega = kwargs["smooth_opts"]["omega"]
                # l_plus = np.asarray([0, -2.0/omega, 0])
                # l_minus = np.asarray([1.0, -2.0*(1.0 - 1.0/omega), 1.0])
                l_plus = self.stencils[-1].l_plus_jacobi(omega)
                l_minus = self.stencils[-1].l_minus_jacobi(omega)
                self.smoothers.append(SplitSmoother(l_plus, l_minus, self.levels[-1]))
            elif kwargs["smoothing_type"] is "ilu":
                self.smoothers.append(ILUSmoother(self.stencils[-1], self.levels[-1], **kwargs["smooth_opts"]))
            else:
                raise ValueError("Wrong smoothing type")
            # append interpolation

            self.ipl_ops.append(kwargs["ipl_class"](self.levels[-2], self.levels[-1],
                                                    *kwargs.get("ipl_opts"), pre_assign=iadd))
            self.rst_ops.append(kwargs["rst_class"](self.levels[-1], self.levels[-2],
                                                    *kwargs.get("rst_opts")))

    def set_initial_value(self, lvl_ind, data):
        self.levels[lvl_ind].mid[:] = data

    def pad(self, lvl_ind):
        self.levels[lvl_ind].pad()

    def modify_rhs(self, ind):
        self.stencils[ind].modify_rhs(self.levels[ind])

    def fill_rhs(self, ind):
        self.mg_problem.fill_rhs(self.levels[ind])

    def run(self, controlflow):
        print(controlflow)
        print("Starting calculation . . .")

    def v_cycle_verbose(self):
        # start with top_level down the v_cycle

        for i in range(1, self.num_levels):
            k = self.num_levels - i + 1
            print("Level %d before smoothing" % k)
            self.levels[-i].print_all()

            self.smoothers[-i].relax(self.n_pre)

            print("Level %d after smoothing" % k)
            self.levels[-i].print_all()

            self.levels[-i].compute_residual(self.stencils[-i])

            print("Level %d after residual computation" % k)
            self.levels[-i].print_all()

            self.rst_ops[-i].restrict()
        # compute direct solution on coarsest level
        print("Level %d before direct solve" % 0)
        self.levels[0].print_all()

        self.smoothers[0].relax()

        print("Level %d after direct solve" % 0)
        self.levels[0].print_all()

        # up the v cycle
        for i in range(self.num_levels-1):

            self.ipl_ops[i].eval()

            print("Level %d after residual computation" % (i+1))
            self.levels[i+1].print_all()

            self.smoothers[i+1].relax(self.n_post)

            print("Level %d after smoothing" % (i+1))
            self.levels[i+1].print_all()

    def v_cycle(self):
        # start with top_level down the v_cycle

        for i in range(1, self.num_levels):
            self.smoothers[-i].relax(self.n_pre)
            self.levels[-i].compute_residual(self.stencils[-i])
            self.rst_ops[-i].restrict()

        # compute direct solution on coarsest level

        self.smoothers[0].relax()


        # up the v cycle
        for i in range(self.num_levels-1):
            self.ipl_ops[i].eval()
            self.smoothers[i+1].relax(self.n_post)


    def w_cycle(self, max_depth):
        pass

    def fmg_cycle(self, max_depth):
        pass

if __name__ == '__main__':
    laplace_array = np.asarray([1.0, -2.0, 1.0])
    laplace_stencil = Stencil(np.asarray([1, -2, 1]), None, 2)

    # preparing MG_Problem

    geo = np.asarray([[0, 1]])
    boundary_type = ["dirichlet"]*2
    left_f = lambda x: 0
    right_f = lambda x: 1.0
    boundary_functions = [[left_f, right_f]]

    def stupid_f(*args, **kwargs):
        return 0.0

    mg_problem = MultiGridProblem(laplace_stencil,
                                  stupid_f,
                                  boundary_functions=boundary_functions,
                                  boundaries=boundary_type, geometry=geo)

    print("mg_problems.boundaries", mg_problem.boundaries)
    def stencil_form_2d(level):
        return (np.asarray([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]) / level.h[0]**2,
                np.asarray([1, 1]))

    def stencil_form_1d(level):
        return (np.asarray([1.0, -2.0, 1.0]) / level.h**2, np.asarray([1]))

    MG_CORE_OPTIONS = {}
    MG_CORE_OPTIONS.update(MG_SMOOTHER_PRESETS["Jacobi"])
    MG_CORE_OPTIONS.update(MG_LEVEL_PRESETS["Standard-1D"])
    MG_CORE_OPTIONS.update(MG_RESTRICTION_PRESETS["Standard-1D"])
    MG_CORE_OPTIONS.update(MG_INTERPOLATION_PRESETS["Standard-1D"])
    MG_CORE_OPTIONS["shape_coarse"] = 2
    MG_CORE_OPTIONS["n_pre"] = 1
    MG_CORE_OPTIONS["n_post"] = 1
    mg_core = MultiGridCore(mg_problem, stencil_form_1d, **MG_CORE_OPTIONS)
    mg_core.fill_rhs(-1)
    mg_core.pad(-1)
    mg_core.modify_rhs(-1)
    mg_core.v_cycle()
    # mg_core.v_cycle_verbose()
    print("After 1 V-Cycle\n", mg_core.levels[-1].mid)
    # plt.imshow(mg_core.levels[-1].arr, cmap=plt.cm.coolwarm)
    # plt.title('Sinus Randbedingungen')
    # plt.xticks([]); plt.yticks([])
    # plt.colorbar(cmap=plt.cm.coolwarm)
    # plt.show()
