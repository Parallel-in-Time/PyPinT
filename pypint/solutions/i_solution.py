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
        self._points = np.zeros(0, dtype=np.float64)
        self._values = np.zeros(0, dtype=np.float64)
        self._errors = np.zeros(0, dtype=np.float64)
        self._residuals = np.zeros(0, dtype=np.float64)
        self._used_iterations = None
        self._reduction = None

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
        if not isinstance(points, np.ndarray) or not isinstance(values, np.ndarray):
            raise ValueError(func_name(self) +
                             "Given points or values is not a numpy.ndarray.")
        if points.size != values.size:
            raise ValueError(func_name(self) +
                             "Points and values must have same size.")
        if "error" in kwargs:
            if not isinstance(kwargs["error"], np.ndarray):
                raise ValueError(func_name(self) +
                                 "Given error data is not a numpy.ndarray.")
        if "residual" in kwargs:
            if not isinstance(kwargs["residual"], np.ndarray):
                raise ValueError(func_name(self) +
                                 "Given residual data is not a numpy.ndarray.")

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
        pass

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
        pass

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
        pass

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
    def values(self):
        """
        Summary
        -------
        Accessor for the complete solution data.

        Returns
        -------
        raw solution data : numpy.ndarray
        """
        return self._values

    @property
    def errors(self):
        """
        Summary
        -------
        Accessor for the complete errors data.

        Returns
        -------
        raw errors data : numpy.ndarray
        """
        return self._errors

    @property
    def residuals(self):
        """
        Summary
        -------
        Accessor for the complete residuals data.

        Returns
        -------
        raw residuals data : numpy.ndarray
        """
        return self._residuals

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
