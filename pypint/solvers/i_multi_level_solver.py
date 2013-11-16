# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.solvers.i_iterative_time_solver import IIterativeTimeSolver


class IMultiLevelSolver(IIterativeTimeSolver):
    """
    basic interface for iterative multi-time-level solvers
    """

    def __init__(self):
        """

        """
        self._base_solver = None
        self._base_level = -1
        self._top_level = -1
        self._multi_level_provider = None

    def init(self, problem, base_solver, base_level, top_level, multi_level_provider):
        self._base_solver = base_solver
        self._base_level = base_level
        self._top_level = top_level
        self._multi_level_provider = multi_level_provider
        super().init(problem)

    @property
    def base_solver(self):
        return self._base_solver

    @property
    def base_level(self):
        return self._base_level

    @property
    def top_level(self):
        return self._top_level

    @property
    def multi_level_provider(self):
        return self._multi_level_provider
