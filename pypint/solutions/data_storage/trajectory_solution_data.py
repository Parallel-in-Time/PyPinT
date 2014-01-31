# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import warnings

import numpy as np

from pypint.solutions.data_storage.step_solution_data import StepSolutionData
from pypint.utilities import assert_condition


class TrajectorySolutionData(object):
    """
    Summary
    -------
    Storage for a transient trajectory of solutions.

    Extended Summary
    ----------------
    Basically, this is nothing more than an array of :py:class:`.StepSolutionData` objects and a couple of
    utility functions for easy data access and consistency checks.
    """

    def __init__(self):
        # self._data: numpy.ndarray of StepSolutionData instances
        self._data = np.zeros(0, dtype=np.object)
        self._time_points = np.zeros(0, dtype=np.float)
        self._numeric_type = None
        self._dim = None

    def add_solution_data(self, *args, **kwargs):
        """
        Summary
        -------
        Appends solution of a new time point to the trajectory.

        Parameters
        ----------
        step_data : :py:class:`.StepSolutionData`
            *(optional)*
            In case a single unnamed argument is given, this is required to be an instance of
            :py:class:`.StepSolutionData`.
            If no named argument is given, the following two parameters are *not* optional.

        values : :py:class:`numpy.ndarray`
            *(optional)*
            Solution values.
            Passed on to constructor of :py:class:`.StepSolutionData`.

        time_point : :py:class:`float`
            *(optional)*
            Time point of the solution.
            Passed on to constructor of :py:class:`.StepSolutionData`.

        Raises
        ------
        ValueError :
            * if construction of :py:class:`.StepSolutionData` fails
            * if internal consistency check fails (see :py:meth:`._check_consistency`)
        """
        _old_data = self._data  # backup for potential rollback

        if len(args) == 1 and isinstance(args[0], StepSolutionData):
            self._data = np.append(self._data, np.array([args[0]], dtype=np.object))
        else:
            self._data = np.append(self._data, np.array([StepSolutionData(*args, **kwargs)], dtype=np.object))

        try:
            self._check_consistency()
        except ValueError as err:
            # consistency check failed, thus removing recently added solution data storage
            warnings.warn("Consistency Check failed with:\n\t\t{}\n\tNot adding this solution.".format(*err.args))
            self._data = _old_data.copy()  # rollback
            raise err
        finally:
            # everything ok
            pass

        if self._data.size == 1:
            self._dim = self._data[-1].dim
            self._numeric_type = self._data[-1].numeric_type

    @property
    def data(self):
        """
        Summary
        -------
        Read-only accessor for the stored solution objects.

        Returns
        -------
        data : :py:class:`numpy.ndarray` of :py:class:`.StepSolutionData`
        """
        return self._data

    @property
    def time_points(self):
        """
        Summary
        -------
        Accessor for the time points of stored solution data.

        Returns
        -------
        error : :py:class:`numpy.ndarray` of :py:class:`float`
        """
        return np.array([step.time_point for step in self.data])

    @property
    def values(self):
        """
        Summary
        -------
        Accessor for the solution values of stored solution data.

        Returns
        -------
        error : :py:class:`numpy.ndarray` of :py:class:`.numeric_type`
        """
        return np.array([step.value for step in self.data])

    @property
    def errors(self):
        """
        Summary
        -------
        Accessor for the errors of stored solution data.

        Returns
        -------
        error : :py:class:`numpy.ndarray` of :py:class:`.Error`
        """
        return np.array([step.error for step in self.data])

    @property
    def residuals(self):
        """
        Summary
        -------
        Accessor for the residuals of stored solution data.

        Returns
        -------
        error : :py:class:`numpy.ndarray` of :py:class:`.Residual`
        """
        return np.array([step.residual for step in self.data])

    @property
    def numeric_type(self):
        """
        Summary
        -------
        Read-only accessor for the numeric type of the solution data values.
        """
        return self._numeric_type

    @property
    def dim(self):
        """
        Summary
        -------
        Read-only accessor for the spacial dimension of the solution data values.
        """
        return self._dim

    def _check_consistency(self):
        """
        Summary
        -------
        Checks for consistency of spacial dimension and numeric type of stored steps.

        Raises
        ------
        ValueError :
            * if the numeric type of at least one step does not match :py:attr:`.numeric_type`
            * if the spacial dimension of at least one step does not match :py:attr:`.dim`
        """
        if self._data.size > 0:
            _time_point = self.data[0].time_point
            for step in range(1, self.data.size):
                assert_condition(self.data[step].time_point > _time_point,
                                 ValueError, "Time points must be strictly increasing: {:f} <= {:f}"
                                             .format(self.data[step].time_point, _time_point),
                                 self)
                assert_condition(self.data[step].numeric_type == self.numeric_type,
                                 ValueError,
                                 ("Numeric type of step {:d} does not match global numeric type: "
                                  .format(step, self.numeric_type) +
                                  "{} != {}".format(self.data[step].numeric_type, self.numeric_type)),
                                 self)
                assert_condition(self.data[step].dim == self.dim,
                                 ValueError,
                                 ("Spacial dimension of step {:d} does not match global spacial dimension: "
                                  .format(step, self.dim) + "{:d} != {:d}".format(self.data[step].dim, self.dim)),
                                 self)

    def append(self, p_object):
        """
        See Also
        --------
        .add_solution_data : with one unnamed parameter
        """
        self.add_solution_data(p_object)

    def __len__(self):
        return self._data.size

    def __getitem__(self, item):
        return self._data[item]

    def __delitem__(self, key):
        del self._data[key]

    def __setitem__(self, key, value):
        self.add_solution_data(value=value, time_point=key)

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, item):
        assert_condition(isinstance(item, StepSolutionData),
                         TypeError, "Item must be a StepSolutionData: NOT {}".format(item.__class__.__name__),
                         self)
        for elem in self._data:
            if elem == item:
                return True
        return False


__all__ = ['TrajectorySolutionData']
