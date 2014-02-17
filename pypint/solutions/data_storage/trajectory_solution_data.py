# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import warnings

import numpy as np

from pypint.solutions.data_storage.step_solution_data import StepSolutionData
from pypint.utilities import assert_condition


class TrajectorySolutionData(object):
    """Storage for a transient trajectory of solutions.

    Basically, this is nothing more than an array of :py:class:`.StepSolutionData` objects and a couple of
    utility functions for easy data access and consistency checks.

    This class provides a selected subset of Python's mutable sequence datatype methods:

    :``__len__``:
        Returns the number of :py:class:`.StepSolutionData` objects stored in this instance.

    :``__gettiem__``:
        Takes the 0-based index of the :py:class:`.StepSolutionData` object to query.

    :``__setitem__``:
        Takes a :py:class:`float` representing the time point and a :py:class:`numpy.ndarray` as the rvalue.
        Same as ``.add_solution_data(value=<rvalue>, time_point=<time_point>)``.

    :``__iter__``:
        Gives an iterator over the stored :py:class:`.StepSolutionData` objects (proxies
        :py:meth:`numpy.ndarray.__iter__`).

    :``__contains__``:
        Finds the given :py:class:`.StepSolutionData` object in this sequence.
    """

    def __init__(self):
        # self._data: numpy.ndarray of StepSolutionData instances
        self._data = np.zeros(0, dtype=np.object)
        self._time_points = np.zeros(0, dtype=np.float)
        self._numeric_type = None
        self._dim = None
        self._finalized = False

    def add_solution_data(self, *args, **kwargs):
        """Appends solution of a new time point to the trajectory.

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
        assert_condition(not self.finalized, AttributeError, "Cannot change this solution data storage any more.", self)
        _old_data = self._data  # backup for potential rollback

        if len(args) == 1 and isinstance(args[0], StepSolutionData):
            assert_condition(args[0].time_point is not None,
                             ValueError, "Time point must not be None.",
                             self)
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

    def finalize(self):
        """Locks this storage data instance.

        Raises
        ------
        ValueError :
            If it has already been locked.
        """
        assert_condition(not self.finalized, AttributeError, "This solution data storage is already finalized.", self)
        self._finalized = True

    @property
    def finalized(self):
        """Accessor for the lock state.

        Returns
        -------
        finilized : :py:class:`bool`

            :``True``:
                if it has been finalized before
            :``False``:
                otherwise
        """
        return self._finalized

    @property
    def data(self):
        """Read-only accessor for the stored solution objects.

        Returns
        -------
        data : :py:class:`numpy.ndarray` of :py:class:`.StepSolutionData`
        """
        return self._data

    @property
    def time_points(self):
        """Accessor for the time points of stored solution data.

        Returns
        -------
        error : :py:class:`numpy.ndarray` of :py:class:`float`
        """
        return np.array([step.time_point for step in self.data], dtype=np.float)

    @property
    def values(self):
        """Accessor for the solution values of stored solution data.

        Returns
        -------
        error : :py:class:`numpy.ndarray` of :py:class:`.numeric_type`
        """
        return np.array([step.value for step in self.data], dtype=np.object)

    @property
    def errors(self):
        """Accessor for the errors of stored solution data.

        Returns
        -------
        error : :py:class:`numpy.ndarray` of :py:class:`.Error`
        """
        return np.array([step.error for step in self.data], dtype=np.object)

    @property
    def residuals(self):
        """Accessor for the residuals of stored solution data.

        Returns
        -------
        error : :py:class:`numpy.ndarray` of :py:class:`.Residual`
        """
        return np.array([step.residual for step in self.data], dtype=np.object)

    @property
    def numeric_type(self):
        """Read-only accessor for the numeric type of the solution data values.
        """
        return self._numeric_type

    @property
    def dim(self):
        """Read-only accessor for the spacial dimension of the solution data values.
        """
        return self._dim

    def _check_consistency(self):
        """Checks for consistency of spacial dimension and numeric type of stored steps.

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

    def __str__(self):
        return "TrajectorySolutionData(data={}, time_points={})".format(self.data, self.time_points)


__all__ = ['TrajectorySolutionData']
