# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import warnings
import copy

import numpy as np

from pypint.solutions.i_solution import ISolution
from pypint.solutions.data_storage.step_solution_data import StepSolutionData
from pypint.solutions.data_storage.trajectory_solution_data import TrajectorySolutionData
from pypint.utilities import assert_is_instance, assert_condition


class IterativeSolution(ISolution):
    """
    Summary
    -------
    Storage for the solutions of an iterative solver.

    Extended Summary
    ----------------
    A new solution of a specific iteration can be added via :py:meth:`.add_solution` and queried via
    :py:meth:`.solution`.

    Examples
    --------
    By default, the internal solution data storage type is :py:class:`.TrajectorySolutionData` allowing for
    storage of the whole trajectory of the solution over the course of several iterations.
    However, this can be changed on initialization to only store the solution of the last time point over the course of
    iterations:

    >>> from pypint.solutions.iterative_solution import IterativeSolution
    >>> from pypint.solutions.data_storage.step_solution_data import StepSolutionData
    >>> my_reduced_full_solution = IterativeSolution(solution_data_type=StepSolutionData)
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        solution_data_type : :py:class:`.TrajectorySolutionData` or :py:class:`.StepSolutionData`
            Defaults to :py:class:`.TrajectorySolutionData`.
        """
        super(IterativeSolution, self).__init__(*args, **kwargs)
        self._data = []
        # As this solution stores all values of all nodes of one iteration, `TrajectorySolutionData` is the solution
        # data type.
        self._data_type = kwargs['solution_data_type'] if 'solution_data_type' in kwargs else TrajectorySolutionData
        self._error_reduction = {}
        self._solution_reduction = {}

    def add_solution(self, *args, **kwargs):
        """
        Summary
        -------
        Adds a new solution data storage object.

        Extended Summary
        ----------------
        After each call an internal consistency check is carried out, which might raise further exceptions.
        The number of used iterations (see :py:attr:`.used_iterations`) is auto-incremented on success.

        Parameters
        ----------
        data : :py:class:`.TrajectorySolutionData` or :py:class:`.StepSolutionData`
            *(must not be named)*
            Exactly one unnamed argument must be given.

        iteration : :py:class:`int`
            *(optional)*
            1-based index of the iteration.
            Defaults to `-1` meaning append after last stored iteration.

        Raises
        ------
        ValueError :
            * if ``iteration`` is not an integer
            * if ``iteration`` is not a valid index for the current size of stored solution data objects
            * if not exactly one solution data object is given
        """
        if 'iteration' in kwargs:
            assert_is_instance(kwargs['iteration'], int,
                               "Iteration index must be an integer: NOT {:s}"
                               .format(kwargs['iteration'].__class__.__name__),
                               self)
            _iteration = kwargs['iteration'] - 1
            if _iteration > 0:
                assert_condition(_iteration in range(-1, self._data.size),
                                 ValueError,
                                 ("Iteration index must be within the size of the solution data array:" +
                                  "{:d} not in [0, {:d}]".format(_iteration, self._data.size)),
                                 self)
            # remove the `iteration` key from the keyword arguments so it does not get passed onto the solution data
            # storage creation
            del kwargs['iteration']
        else:
            _iteration = -1

        assert_condition(len(args) == 1 or 'data' in kwargs,
                         ValueError, "Exactly one solution data object or 'data' must be given.",
                         self)
        assert_is_instance(args[0], self._data_type,
                           "Given solution data storage must be a {}: NOT {}"
                           .format(self._data_type, args[0].__class__.__name__),
                           self)

        _old_data = copy.copy(self._data)  # backup for potential rollback
        if _iteration == -1:
            self._data.append(args[0])
        else:
            self._data.insert(_iteration, args[0])

        try:
            self._check_consistency()
        except ValueError:
            # consistency check failed, thus removing recently added solution data storage
            warnings.warn("Consistency Check failed. Not adding this solution.")
            self._data = copy.copy(_old_data)  # rollback
        finally:
            # everything ok
            pass

        self._used_iterations += 1

    def solution(self, iteration):
        """
        Summary
        -------
        Accessor for the solution of a specific iteration.

        Parameters
        ----------
        iteration : :py:class:`int`
            0-based index of the iteration.
            ``-1`` means last iteration.

        Returns
        -------
        solution : instance of :py:attr:`.data_storage_type`
            or :py:class:`None` if no solutions are stored.

        Raises
        ------
        ValueError :
            if given ``iteration`` index is not in the valid range
        """
        if len(self._data) > 0:
            assert_condition(iteration in range(-1, len(self._data)),
                             ValueError, "Iteration index not within valid range: {:d} not in [-1, {:d}"
                                         .format(iteration, len(self._data)),
                             self)
            return self._data[iteration]
        else:
            return None

    def error(self, iteration):
        """
        Summary
        -------
        Accessor for the errors of a specific iteration.

        Parameters
        ----------
        iteration : :py:class:`int`
            0-based index of the iteration.
            ``-1`` means last iteration.

        Returns
        -------
        error : instance of :py:attr:`.Error`
            or :py:class:`None` if no solutions are stored.

        Raises
        ------
        ValueError :
            if given ``iteration`` index is not in the valid range
        """
        if len(self._data) > 0:
            assert_condition(iteration in range(-1, len(self._data)),
                             ValueError, "Iteration index not within valid range: {:d} not in [-1, {:d}"
                                         .format(iteration, len(self._data)),
                             self)
            if self._data_type == StepSolutionData:
                return np.array(self._data[iteration].error, dtype=np.object)
            else:
                return self._data[iteration].errors
        else:
            return None

    def residual(self, iteration):
        """
        Summary
        -------
        Accessor for the residuals of a specific iteration.

        Parameters
        ----------
        iteration : :py:class:`int`
            0-based index of the iteration.
            ``-1`` means last iteration.

        Returns
        -------
        residual : instance of :py:attr:`.Residual`
            or :py:class:`None` if no solutions are stored.

        Raises
        ------
        ValueError :
            if given ``iteration`` index is not in the valid range
        """
        if len(self._data) > 0:
            assert_condition(iteration in range(-1, len(self._data)),
                             ValueError, "Iteration index not within valid range: {:d} not in [-1, {:d}"
                                         .format(iteration, len(self._data)),
                             self)
            if self._data_type == StepSolutionData:
                return np.array(self._data[iteration].residual, dtype=np.object)
            else:
                return self._data[iteration].residuals
        else:
            return None

    def error_reduction(self, iteration):
        if iteration in self._error_reduction:
            return self._error_reduction[iteration]
        else:
            return None

    def solution_reduction(self, iteration):
        if iteration in self._solution_reduction:
            return self._solution_reduction[iteration]
        else:
            return None

    def set_error_reduction(self, iteration, reduction):
        assert_condition(isinstance(iteration, int) and iteration > 0,
                         ValueError, "Iteration must be a non-zero positive integer: NOT {}".format(iteration),
                         self)
        assert_is_instance(reduction, (float, np.ndarray),
                           "Reduction of error must be given as a float or numpy.ndarray: NOT {}"
                           .format(reduction.__class__.__name__),
                           self)
        self._error_reduction[iteration] = copy.copy(reduction)

    def set_solution_reduction(self, iteration, reduction):
        assert_condition(isinstance(iteration, int) and iteration > 0,
                         ValueError, "Iteration must be a non-zero positive integer: NOT {}".format(iteration),
                         self)
        assert_is_instance(reduction, (float, np.ndarray),
                           "Reduction of solution must be given as a float or numpy.ndarray: NOT {}"
                           .format(reduction.__class__.__name__),
                           self)
        self._solution_reduction[iteration] = copy.copy(reduction)

    @property
    def solutions(self):
        """
        Summary
        -------
        Read-only accessor for the stored list of solution data storages.

        Returns
        -------
        values : :py:class:`list` of :py:class:`.TrajectorySolutionData` or :py:class:`.StepSolutionData` objects
        """
        return self._data

    @property
    def time_points(self):
        """
        Summary
        -------
        Proxies :py:attr:`.TrajectorySolutionData.time_points`.

        Returns
        -------
        time_points : :py:class:`numpy.ndarray` or :py:class:`None`
            :py:class:`None` is returned if no solutions have yet been stored
        """
        if len(self._data) > 0:
            if self._data_type == TrajectorySolutionData:
                return self._data[0].time_points
            else:
                return np.array([step.time_point for step in self._data], dtype=np.float)
        else:
            return None

    def _check_consistency(self):
        """
        Summary
        -------
        Check consistency of stored solution data objects.

        Raises
        ------
        ValueError :
            * if the time points of at least two solution data storage objects differ
        """
        if len(self._data) > 0:
            _time_points = self._data[0].time_points
            for iteration in range(1, len(self._data)):
                assert_condition(np.array_equal(_time_points, self._data[iteration].time_points),
                                 ValueError, "Time points of one or more stored solution data objects do not match.",
                                 self)


__all__ = ['IterativeSolution']
