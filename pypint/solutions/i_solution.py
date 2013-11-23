# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

import numpy as np
from pypint.utilities import func_name


class ISolution(object):
    """
    Summary
    -------
    Generalized storage for solutions of solvers.
    """
    def __init__(self):
        self._data = np.zeros(0, dtype=np.float64)
        self._errors = np.zeros(0, dtype=np.float64)
        self._used_iterations = None
        self._reduction = None

    def add_solution(self, data, *args, **kwargs):
        """
        Summary
        -------
        Adds a new solution of the specified iteration.

        Parameters
        ----------
        data : numpy.ndarray
             Solution data.

        args : list
            Further arguments not processed by this class.

        kwargs : dict
            Further named arguments not processed by this class.

        Raises
        ------
        ValueError
            If either ``data`` is not a ``numpy.ndarray``
        """
        if not isinstance(data, np.ndarray):
            raise ValueError(func_name(self) +
                             "Given data is not a numpy.ndarray.")
        if "error" in kwargs:
            if not isinstance(kwargs["error"], np.ndarray):
                raise ValueError(func_name(self) +
                                 "Given error data is not a numpy.ndarray.")

    def solution(self, *args, **kwargs):
        """
        Summary
        -------
        Accessor for a specific solution

        Extended Summary
        ----------------
        Should be overridden by derived classes if applicable.

        Parameters
        ----------
        args : list
            Not processed (unnamed) arguments.

        kwargs : dict
            Descriptors (named arguments) of the desired solution.

        Returns
        -------
        Nothing
        """
        pass

    def error(self, *args, **kwargs):
        pass

    @property
    def data(self):
        """
        Summary
        -------
        Accessor for the complete solution data.

        Returns
        -------
        raw solution data : numpy.ndarray
        """
        return self._data

    @property
    def errors(self):
        return self._errors

    @property
    def used_iterations(self):
        """
        Summary
        -------
        Accessor for the number of iterations.

        Parameters
        ----------
        used_levels : integer
            Number of used levels to be set.

        Returns
        -------
        used levels : integer
            Number of used levels.
        """
        return self._used_iterations

    @used_iterations.setter
    def used_iterations(self, used_iterations):
        self._used_iterations = int(used_iterations)

    @property
    def reduction(self):
        """
        Summary
        -------
        Accessor for the overall reduction of the solver.

        Parameters
        ----------
        reduction : float
            Reduction to be set.

        Returns
        -------
        reduction : float
        """
        return self._reduction

    @reduction.setter
    def reduction(self, reduction):
        self._reduction = reduction

    def __str__(self):
        return self.__class__.__name__ + ": {:s}".format(self._data)
