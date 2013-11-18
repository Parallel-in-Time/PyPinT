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
        super(self.__class__, self).__init__()
        self._data = np.array()
        self._used_iterations = 0
        self._reduction = 0.0

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
        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("pypint.solutions.SimpleSolution.data():" +
                             "Given data is not a numpy.ndarray.")
        self._data = data
