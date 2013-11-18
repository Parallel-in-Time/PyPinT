# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class IIterativeTimeSolver(object):
    """
    basic interface for iterative time solvers
    """

    def __init__(self):
        """

        """
        self._problem = None
        self._logger = None
        self._timer = None
        self._max_iterations = -1
        self._min_reduction = -1

    def init(self, problem, *args, **kwargs):
        self._problem = problem

    def run(self):
        return None

    @property
    def problem(self):
        return self._problem

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger):
        self._logger = logger

    @property
    def timer(self):
        return self._timer

    @timer.setter
    def timer(self, timer):
        self._timer = timer

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, max_iterations):
        self._max_iterations = max_iterations

    @property
    def min_reduction(self):
        return self._min_reduction

    @min_reduction.setter
    def min_reduction(self, min_reduction):
        self._min_reduction = min_reduction
