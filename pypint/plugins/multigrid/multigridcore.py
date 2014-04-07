# coding=utf-8
import numpy as np
from pypint.plugins.multigrid.multigridproblem import MultiGridProblem
from pypint.plugins.multigrid.multigridlevelprovider import MultiGridLevelProvider
from pypint.plugins.multigrid.multigridsolution import MultiGridSolution
from pypint.plugins.multigrid.level import MultiGridLevel1D
from pypint.plugins.multigrid.multigrid_smoother import SplitSmoother, DirectSolverSmoother, WeightedJacobiSmoother
from pypint.utilities import assert_is_callable, assert_is_instance, assert_condition
from pypint.plugins.multigrid.stencil import Stencil
import networkx as nx

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

    def __init__(self, mg_prob, mg_level_prov, mg_solution, mg_control):
        # Check types
        assert_is_instance(mg_prob, MultiGridProblem,
                           "not a proper Multigridproblem", self)
        assert_is_instance(mg_level_prov, MultiGridLevelProvider,
                           "not a proper multigridlevelprovider", self)
        assert_is_instance(mg_solution, MultiGridSolution,
                           "not a proper multigridsolution", self)
        self.mg_prob = mg_prob
        self.mg_level_prov = mg_level_prov
        self.mg_solution = mg_solution
        self.mg_control = mg_control


    def run(self, controlflow):
        print(controlflow)
        print("Starting calculation . . .")

    def v_cycle(self, max_depth):
        pass

    def w_cycle(self, max_depth):
        pass

    def fmg_cycle(self, max_depth):
        pass

    def jacobi_smoother(self, omega):
        pass

    def gauss_seidel_smoother(self):
        pass

if __name__ == '__main__':
    print("Lets solve, you guessed it, the heat equation")
    # heat equation needs a stencil
    laplace_stencil = Stencil(np.asarray([1, -2, 1]))
    # test stencil to some extend
    print("===== Stencil tests =====")
    print("stencil.b :", laplace_stencil.b)
    print("stencil.positions :", laplace_stencil.positions)

    for i in laplace_stencil.positions:
        print(laplace_stencil.arr[i])

    print("stencil.relative_position :", laplace_stencil.relative_positions)
    print("stencil.relative_position_woc :", laplace_stencil.relative_positions_woc)
    # geometry is a 1 dimensional line
    geo = np.asarray([[0, 1]])
    print(geo.shape)
    # the boundary conditions, in this case dirichlet boundary conditions
    boundary_type = ["dirichlet"]*2
    left_f = lambda x: 100.0
    right_f = lambda x: 110.0
    boundary_functions = [[left_f, right_f]]
    rhs_function = lambda x: 0.0
    mg_problem = MultiGridProblem(laplace_stencil,
                                  rhs_function,
                                  boundary_functions=boundary_functions,
                                  boundaries=boundary_type,
                                  geometry=geo)
    # test some of the methods of mg_problem

    print("Mid of stencil method", mg_problem.mid_of_stencil(laplace_stencil))
    print("Eval_Convolve ([100,105,105,105,110])", laplace_stencil.eval_convolve(np.asarray([100, 105, 105, 105, 110])))
    print("===== MultiGridProblemTest =====")
    print("Constructed SpaceTensor", mg_problem.construct_space_tensor(12))
    print("Checked if the grid distances are right",
          mg_problem.act_grid_distances)

    # they work properly at least for the 1d case
    # lets define the different levels lets try 3
    borders = np.asarray([3, 3])

    top_level = MultiGridLevel1D(512, mg_problem=mg_problem,
                                 max_borders=borders)

    mid_level = MultiGridLevel1D(256, mg_problem=mg_problem,
                                 max_borders=borders)

    low_level = MultiGridLevel1D(64, mg_problem=mg_problem,
                                 max_borders=borders)
    # check if the distance between points is calculated right
    print("===== MultiGridLevel Test =====")
    print("3 different GridDistances from top to low level:")
    print(top_level.h)
    print(mid_level.h)
    print(low_level.h)
    print("the space_tensor of the last level")
    print(low_level.space_tensor)
    print(*mg_problem.act_grid_distances)
    print("test the is on border function")
    is_on_border = low_level.border_function_generator(laplace_stencil)
    border_truth = [is_on_border((i,)) for i in range(low_level.evaluable_view(laplace_stencil).size)]
    print(border_truth)

    # define the smoother from the split smoother class on each level,
    # where the last level is solved directly
    l_plus = np.asarray([0, -2, 0])
    l_minus = np.asarray([1, 0, 1])
    omega = 0.5
    top_jacobi_smoother = SplitSmoother(l_plus / top_level.h / omega,
                                        l_minus / top_level.h * (1 - 1 / omega),
                                        top_level)
    mid_jacobi_smoother = SplitSmoother(l_plus / mid_level.h / omega,
                                        l_minus / mid_level.h * (1 - 1 / omega),
                                        mid_level)
    low_direct_smoother = DirectSolverSmoother(laplace_stencil, low_level)
    # time to test the relaxation methods
    print("===== DirectSolverSmoother Test =====")
    low_level.rhs[:] = 0.0
    low_level.pad()
    print("arr:", low_level.arr)
    laplace_stencil.modify_rhs(low_level)
    print("rhs:", low_level.rhs)
    low_direct_smoother.relax()
    print(low_level.arr)
    low_level.pad()
    # Lets test the SplitSmoother by using the jacobi smoother
    # but for this case we need an initial guess
    print("===== JacobiSmoother Test =====")
    # define the 3 different JacobiSmoother Implementations
    jacobi_loop = WeightedJacobiSmoother(laplace_stencil,
                                         low_level, 0.5, "loop")
    jacobi_matrix = WeightedJacobiSmoother(laplace_stencil,
                                           low_level, 0.5, "matrix")
    jacobi_convolve = WeightedJacobiSmoother(laplace_stencil,
                                            low_level, 0.5, "convolve")

    low_level.mid[:] = 105.0
    low_level.pad()
    print("We start with this initial condition:")
    print(low_level.arr)
    print("Now we do a jacobi step using the convolve of stencil:")
    # funktioniert beinahe ist nur falsch rum
    jacobi_convolve.relax()
    print(low_level.arr)
    low_level.mid[:] = 105.0
    low_level.pad()
    # print("Wurde hier der stencil veraendert?", laplace_stencil.arr)
    print("Now we do a jacobi step using simple loops:")
    jacobi_loop.relax()
    print(low_level.arr)
    low_level.mid[:] = 105.0
    low_level.pad()

    print("Now we do a jacobi step using sparse matrix algorithms:")

    jacobi_matrix.relax()
    print(low_level.arr)


    #mid_level.mid[:] = 105
    #mid_jacobi_smoother.relax(5)
    #print(mid_level.mid[:])


