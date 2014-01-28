# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_solution_data import ISolutionData
from ..utilities import assert_is_instance
import numpy as np
import warnings


class StepSolutionData(ISolutionData):
    """
    Summary
    -------
    Storage for the solution of a single time point.

    Notes
    -----
    It should not be necessary to directly and explicitly create instances of this class.
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        value : :py:class:`numpy.ndarray`
            Solution value.
            Size of the vector must equal the spacial dimension of the problem.

        time_point : :py:class:`float`
            Time point of associated with the solution.

        error : :py:class:`numpy.ndarray`
            Error of the solution.
            There are no constrains on the size of the vector, thus both, a general :math:`\\mathcal{L}^2` error or
            component-wise absolute error can be stored.

        residual : :py:class:`numpy.ndarray`
            Residual of the solution.
            Same abundance of constrains apply as for the error.

        Raises
        ------
        ValueError :
            * if `value`, `error` or `residual` is not a :py:class:`numpy.ndarray`
            * if `time_point` is not a :py:class:`float`

        UserWarning :
            * if no `value` or `time_point` is given

        Notes
        -----
        The spacial dimension and the numerical type are derived from the given solution values.
        Thus, specifying :py:attr:`.solutions.ISolutionData.dim` and :py:attr:`.solutions.ISolutionData.numeric_type`
        is not recommended (in fact they are ignored).
        """
        super(StepSolutionData, self).__init__(*args, **kwargs)

        if "value" in kwargs:
            assert_is_instance(kwargs["value"], np.ndarray,
                               "Values must be a numpy.ndarray: NOT {:s}".format(kwargs["value"].__class__.__name__),
                               self)
            self._data = kwargs["value"].copy()
            self._dim = self._data.size
            self._numeric_type = self._data.dtype
        else:
            warnings.warn("No solution value given.")

        self._time_point = None
        if "time_point" in kwargs:
            assert_is_instance(kwargs["time_point"], float,
                               "Time point must be a float: NOT {:s}".format(kwargs["time_point"].__class__.__name__),
                               self)
            self._time_point = kwargs["time_point"]
        else:
            warnings.warn("No time point for the solution given.")

        self._error = None
        if "error" in kwargs:
            assert_is_instance(kwargs["error"], np.ndarray,
                               "Error must be a numpy.ndarray: NOT {:s}".format(kwargs["error"].__class__.__name__),
                               self)
            self._error = kwargs["error"].copy()

        self._residual = None
        if "residual" in kwargs:
            assert_is_instance(kwargs["residual"], np.ndarray,
                               "Residual must be a numpy.ndarray: NOT {:s}".format(kwargs["residual"].__class__.__name__),
                               self)
            self._residual = kwargs["residual"].copy()

    @property
    def value(self):
        """
        Summary
        -------
        Read-only accessor for the solution value.

        Returns
        -------
        value : :py:class:`numpy.ndarray`
        """
        return self._data

    @property
    def time_point(self):
        """
        Summary
        -------
        Read-only accessor for the associated time point.

        Returns
        -------
        time_point : :py:class:`float`
        """
        return self._time_point

    @property
    def error(self):
        """
        Summary
        -------
        Read-only accessor for the error.

        Returns
        -------
        error : :py:class:`None`
            if no error is given
        """
        return self._error

    @property
    def residual(self):
        """
        Summary
        -------
        Read-only accessor for the residual.

        Returns
        -------
        residual : :py:class:`None`
            if no residual is given
        """
        return self._residual

__all__ = ['StepSolutionData']
