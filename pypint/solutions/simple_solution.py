# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_solution import ISolution
import numpy as np


class SimpleSolution(ISolution):
    """
    storage for the final solution of a solver

    Stores the solution as a numpy.ndarray.
    """
    def __init__(self):
        super().__init__()
        self.__data = np.array()
        self.__used_iterations = 0
        self.__reduction = 0.0

    @property
    def data(self):
        """
        accessing the stored solution

        **Getter**

        :returns: stored solution
        :rtype:   numpy.ndarray

        **Setter**

        :param data: solution to store
        :type data:  numpy.ndarray

        :raises: ValueError if ``data`` is not a ``numpy.ndarray``
        """
        return self.__data

    @data.setter
    def data(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("pypint.solutions.SimpleSolution.data():" +
                             "Given data is not a numpy.ndarray.")
        self.__data = data
