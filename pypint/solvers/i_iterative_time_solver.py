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
        self.__problem = None
        self.__logger = None
        self.__timer = None
        self.__max_iterations = -1
        self.__min_reduction = -1

    def init(self, problem):
        self.__problem = problem

    def run(self):
        return None

    @property
    def problem(self):
        return self.__problem

    @property
    def logger(self):
        return self.__logger

    @logger.setter
    def logger(self, logger):
        self.__logger = logger

    @property
    def timer(self):
        return self.__timer

    @timer.setter
    def timer(self, timer):
        self.__timer = timer

    @property
    def max_iterations(self):
        return self.__max_iterations

    @max_iterations.setter
    def max_iterations(self, max_iterations):
        self.__max_iterations = max_iterations

    @property
    def min_reduction(self):
        return self.__min_reduction

    @min_reduction.setter
    def min_reduction(self, min_reduction):
        self.__min_reduction = min_reduction
