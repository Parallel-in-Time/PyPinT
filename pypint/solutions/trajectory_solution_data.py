# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_solution_data import ISolutionData
from .step_solution_data import StepSolutionData
from ..utilities import assert_condition, assert_is_instance
import numpy as np
import warnings


class TrajectorySolutionData(ISolutionData):
    """
    Summary
    -------
    Storage for a transient trajectory of solutions.

    Notes
    -----
    It should not be necessary to directly and explicitly create instances of this class.
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        values : :py:class:`numpy.ndarray`
            Vector of :py:class:`.solutions.StepSolutionData` objects of the time points of the solution trajectory.

        Raises
        ------
        ValueError :
            If `values` is not a :py:class:`numpy.ndarray` with :py:class:`numpy.dtype` of `object` and the elements of
            the vector are not all :py:class:`.solutions.StepSolutionData` objects.

        UserWarning :
            If no `values` is given.
        """
        super(TrajectorySolutionData, self).__init__(*args, **kwargs)

        if "values" in kwargs:
            assert_is_instance(kwargs["values"], np.ndarray,
                               "Values must be a numpy.ndarray: NOT {:s}".format(kwargs["values"].__class__.__name__),
                               self)
            assert_condition(kwargs["values"].dtype.kind == 'O',
                             ValueError, "Values must be a numpy.ndarray of objects: NOT {:s}"
                                         .format(kwargs["values"].dtype.kind),
                             self)
            for element in kwargs["values"]:
                assert_is_instance(element, StepSolutionData,
                                   ("The elements of a trajectory solution must all be 'StepSolutionData' objects: " +
                                    "NOT {:s}".format(element.__class__.__name__)), self)

            self._data = kwargs["values"].copy()
            self._dim = self._data[0].dim
            self._numeric_type = self._data[0].numeric_type
        else:
            warnings.warn("No solution values given.")

        self._time_points = None
        self._parse_time_points()
        self._errors = None
        self._parse_errors()
        self._residuals = None
        self._parse_residuals()

    @property
    def values(self):
        """
        Summary
        -------
        Read-only accessor for the stored solution objects.
        """
        return self._data

    @property
    def time_points(self):
        """
        Summary
        -------
        Read-only accessor for the time points of stored solution data.
        """
        return self._time_points

    @property
    def errors(self):
        """
        Summary
        -------
        Read-only accessor for the errors of stored solution data.
        """
        return self._errors

    @property
    def residuals(self):
        """
        Summary
        -------
        Read-only accessor for the residuals of stored solution data.
        """
        return self._residuals

    def _parse_time_points(self):
        """
        Extended Summary
        ----------------
        Extracts stored time points from the :py:class:`.solutions.StepSolutionData` objects in
        :py:attr:`.solutions.TrajectorySolutionData.values`.
        """
        if self._data is not None:
            self._time_points = np.zeros(self._data.size, dtype=np.float)
            for step_index in range(0, self._data.size):
                self._time_points[step_index] = self._data[step_index].time_point

    def _parse_errors(self):
        """
        Extended Summary
        ----------------
        Extracts stored errors from the :py:class:`.solutions.StepSolutionData` objects in
        :py:attr:`.solutions.TrajectorySolutionData.values`.
        """
        if self._data is not None:
            self._errors = np.zeros(self._data.size, dtype=np.ndarray)
            for step_index in range(0, self._data.size):
                self._errors[step_index] = self._data[step_index].error

    def _parse_residuals(self):
        """
        Extended Summary
        ----------------
        Extracts stored residuals from the :py:class:`.solutions.StepSolutionData` objects in
        :py:attr:`.solutions.TrajectorySolutionData.values`.
        """
        if self._data is not None:
            self._residuals = np.zeros(self._data.size, dtype=np.ndarray)
            for step_index in range(0, self._data.size):
                self._residuals[step_index] = self._data[step_index].residual


__all__ = ['TrajectorySolutionData']
