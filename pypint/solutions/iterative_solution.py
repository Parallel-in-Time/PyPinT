# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_solution import ISolution
import numpy as np
from pypint.utilities import *


class IterativeSolution(ISolution):
    """
    storage for the solutions of an iterative solver

    A new solution of a specific iteration can be added via
    :py:func:`.add_solution` and queried via :py:func:`.solution`.
    """
    def __init__(self):
        super().__init__()
        self._data = []
        self._used_iterations = 0
        self._reduction = 0.0

    def solution(self, iteration):
        """
        queries the solution of the given iteration

        :param iteration: index of the desired solution
        :type iteration:  integer
        :returns: solution of the given iteration
        :rtype:   numpy.ndarray
        """
        return self._data[iteration]

    def add_solution(self, iteration, data):
        """
        adds a new solution of the specified iteration

        :param iteration: index of the iteration of this solution
                          (1-based)
        :type iteration:  integer
        :param data:      solution data
        :type data:       numpy.ndarray

        :raises: **ValueError** if either ``data`` is not a ``numpy.ndarray`` or
          there are more than ``iteration`` solutions already stored.

        .. todo: Fill data of skipped iterations when ``iteration`` is not the
                 next iteration to set.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError(func_name(self) +
                             "Given data is not a numpy.ndarray.")
        if len(self._data) >= iteration:
            raise ValueError(func_name(self) +
                             "Data for iteration {:d} is already present."
                             .format(iteration))
        if len(self._data) < iteration - 1:
            # TODO: fill in unused solutions
            raise NotImplementedError(func_name(self) +
                                      "Skipping of solutions not yet implemented.")
        # append the given solution as the last one
        #  due to previous checks it will have the correct index
        self._data.append(data)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError(func_name(self) +
                             "Given data is not a numpy.ndarray.")
        self._data = data
