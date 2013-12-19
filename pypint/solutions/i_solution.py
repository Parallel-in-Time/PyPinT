# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

import numpy as np
from pypint.utilities import assert_is_instance, critical_assert


class ISolution(object):
    """
    Summary
    -------
    Generalized storage for solutions of solvers.
    """

    class IterationData(object):
        def __init__(self):
            self._iteration = None
            self._values = None
            self._errors = None
            self._residuals = None

        def init(self, iteration, values, errors=None, residuals=None, numeric_type=np.float):
            self._iteration = iteration
            self._values = np.array(values, dtype=numeric_type)
            self._errors = errors
            self._residuals = residuals

        @property
        def iteration(self):
            return self._iteration

        @property
        def values(self):
            return self._values

        @property
        def errors(self):
            return self._errors

        @property
        def residuals(self):
            return self._residuals

    def __init__(self, numeric_type=np.float):
        self._numeric_type = numeric_type
        self._points = np.zeros(0, dtype=np.float)
        self._exact = np.zeros(0, dtype=self.numeric_type)
        self._data = ISolution.IterationData()
        self._used_iterations = None
        self._reductions = None

    def add_solution(self, points, values, *args, **kwargs):
        """
        Summary
        -------
        Adds a new solution of the specified iteration.

        Parameters
        ----------
        points : numpy.ndarray
            Time points of the values.

        values : numpy.ndarray
            Solution values.

        error : numpy.ndarray
            (optional)
            Absolute error of the data.

        residual : numpy.ndarray
            (optional)
            Residual of the data.

        Raises
        ------
        ValueError
            * if either ``points``, ``values``, ``error`` or ``residual`` is not a ``numpy.ndarray``
            * if ``points`` and ``values`` are not of same size
        """
        assert_is_instance(points, np.ndarray, "Points must be a numpy.ndarray.", self)
        assert_is_instance(values, np.ndarray, "Values must be a numpy.ndarray.", self)
        critical_assert(points.size != 0, ValueError, "Number of points must be positive.", self)
        critical_assert(points.size == values.size, ValueError, "Points and values must have same size.", self)
        if "error" in kwargs:
            assert_is_instance(kwargs["error"], np.ndarray, "Error data must be a numpy.ndarray.", self)
        if "residual" in kwargs:
            assert_is_instance(kwargs["residual"], np.ndarray, "Residual data must be a numpy.ndarray.", self)

        if self._points.size == 0:
            self._points = points
        if self._exact.size == 0 and "exact" in kwargs:
            self._exact = kwargs["exact"]

    def solution(self, *args, **kwargs):
        """
        Summary
        -------
        Accessor for a specific solution.

        Extended Summary
        ----------------
        Should be overridden by derived classes if applicable.

        Parameters
        ----------
        kwargs : dict
            Descriptors (named arguments) of the desired solution.

        Returns
        -------
        implementation specific
        """
        return self._data.values

    def exact(self, *args, **kwargs):
        return self._exact

    def error(self, *args, **kwargs):
        """
        Summary
        -------
        Accessor for a specific error.

        Extended Summary
        ----------------
        Should be overridden by derived classes if applicable.

        Parameters
        ----------
        kwargs : dict
            Descriptors (named arguments) of the desired error.

        Returns
        -------
        implementation specific
        """
        return self._data.errors

    def residual(self, *args, **kwargs):
        """
        Summary
        -------
        Accessor for a specific residual.

        Extended Summary
        ----------------
        Should be overridden by derived classes if applicable.

        Parameters
        ----------
        kwargs : dict
            Descriptors (named arguments) of the desired residual.

        Returns
        -------
        implementation specific
        """
        return self._data.residuals

    @property
    def points(self):
        """
        Summary
        -------
        Accessor for all points.

        Returns
        -------
        raw points data : numpy.ndarray
        """
        return self._points

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
    def reductions(self):
        """
        Summary
        -------
        Accessor for the reductions of the solver.

        Parameters
        ----------
        reduction : dict
            Reduction to be set.

        Returns
        -------
        reduction : float
        """
        return self._reductions

    @reductions.setter
    def reductions(self, reductions):
        self._reductions = reductions

    @property
    def numeric_type(self):
        return self._numeric_type

    def __str__(self):
        return self.__class__.__name__ + ": {:s}".format(self._data)
