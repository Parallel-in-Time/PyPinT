# coding=utf-8
import numpy as np
import sys
print(sys.path)
from pypint.plugins.multigrid.multigrid_problem import MultiGridProblem
from pypint.plugins.multigrid.multigrid_level_provider import MultiGridLevelProvider
from pypint.plugins.multigrid.multigrid_solution import MultiGridSolution
from pypint.plugins.multigrid.level import MultigridLevel1D
from pypint.plugins.multigrid.multigrid_smoother import SplitSmoother, DirectSolverSmoother, WeightedJacobiSmoother
from pypint.utilities import assert_is_callable, assert_is_instance, assert_condition
from pypint.plugins.multigrid.stencil import Stencil
from pypint.plugins.multigrid.interpolation import InterpolationByStencilListIn1D, InterpolationByStencilForLevels, InterpolationByStencilForLevelsClassical
from pypint.plugins.multigrid.restriction import RestrictionStencilPure, RestrictionByStencilForLevels, RestrictionByStencilForLevelsClassical
from operator import iadd,add
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


