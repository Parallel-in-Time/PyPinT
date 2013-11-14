# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.solvers.i_iterative_time_solver import IIterativeTimeSolver
from pypint.multi_level_providers.multi_level_provider import *


class IMultiLevelSolver(IIterativeTimeSolver):
    """
    basic interface for iterative multi-time-level solvers
    """

    def __init__(self):
        """

        """
        self.__base_solver = None
        self.__base_level = -1
        self.__top_level = -1
        self.__multi_level_provider = None

    def init(self, problem, base_solver, base_level, top_level, multi_level_provider):
        self.__base_solver = base_solver
        self.__base_level = base_level
        self.__top_level = top_level
        self.__multi_level_provider = multi_level_provider
        super().init(problem)

    @property
    def base_solver(self):
        return self.__base_solver

    @property
    def base_level(self):
        return self.__base_level

    @property
    def top_level(self):
        return self.__top_level

    @property
    def multi_level_provider(self):
        return self.__multi_level_provider
