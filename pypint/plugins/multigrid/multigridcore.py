# coding=utf-8
import numpy as np
from .multigridproblem import MultiGridProblem
from .multigridlevelprovider import MultiGridLevelProvider
from .multigridsolution import MultiGridSolution
from pypint.utilities.tracing import assert_is_callable, assert_is_instance

class ControlMultiGridFlow(object):
    """
    Summary
    _______
    Contains some functions which, for example check if the multigrid is done,
    it also generates the controlflow iterator/string(not sure about it yet).
    gathers the matrices.(Weiss unter anderem ob eine bestimmte matrix fuer ein
    bestimmtes level schon existiert)
    """
    def __init__(self):
        pass

    def am_i_done(self):
        pass

    def next_step(self):
        pass

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

    def __init__(self, mgprob, mglprov, mgsolution, mgcontrol):
        # Check types
        assert_is_instance(mgprob, MultiGridProblem,
                           "not a proper Multigridproblem", self)
        assert_is_instance(mglprov, MultiGridLevelProvider,
                           "not a proper multigridlevelprovider", self)
        assert_is_instance(mgsolution, MultiGridSolution,
                           "not a proper multigridsolution", self)
        self.mgprob = mgprob
        self.mglprov = mglprov
        self.mgsolution = mgsolution
        self.mgcontrol = mgcontrol


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
