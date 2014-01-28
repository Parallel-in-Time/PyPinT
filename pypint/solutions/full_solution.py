# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_solution import ISolution
from .trajectory_solution_data import TrajectorySolutionData
from ..utilities import assert_is_instance, assert_condition
import numpy as np
import warnings


class FullSolution(ISolution):
    """
    Summary
    -------
    Storage for the solutions of an iterative solver.

    Extended Summary
    ----------------
    A new solution of a specific iteration can be added via :py:func:`.add_solution` and queried via
    :py:func:`.solution`.

    Examples
    --------
    By default, the internal solution data storage type is :py:class:`.solutions.TrajectorySolutionData` allowing for
    storage of the whole trajectory of the solution over the course of several iterations.
    However, this can be changed on initialization to only store the solution of the last time point over the course of
    iterations:

    >>> from pypint.solutions import FullSolution, StepSolutionData
    >>> my_full_solution = FullSolution(solution_data_type=StepSolutionData)
    """

    def __init__(self, *args, **kwargs):
        # As this solution stores all values of all nodes of one iteration, `TrajectorySolutionData` is the solution
        # data type.
        super(FullSolution, self).__init__(*args, solution_data_type=TrajectorySolutionData, **kwargs)
        self._data = np.zeros(0, dtype=np.object)

    def add_solution(self, *args, **kwargs):
        """
        Summary
        -------
        Adds a new solution data storage object.

        Extended Summary
        ----------------
        After each call an internal consistency check is carried out, which might raise further exceptions.
        The number of used iterations (see :py:attr:`.solutions.ISolution.used_iterations`) is auto-incremented on
        success.

        Parameters
        ----------
        iteration : :py:class:`int`
            0-based index of the iteration.
            Defaults to `-1` meaning append after last stored iteration.

        Raises
        ------
        ValueError :
            * if `iteration` is not an integer
            * if `iteration` is not a valid index for the current size of stored solution data objects
        """
        if "iteration" in kwargs:
            assert_is_instance(kwargs["iteration"], int,
                               "Iteration index must be an integer: NOT {:s}"
                               .format(kwargs["iteration"].__class__.__name__),
                               self)
            assert_condition(kwargs["iteration"] in range(-1, self._data.size),
                             ("Iteration index must be with the size of the solution data array: {:d} not in [0, {:d}]"
                              .format(kwargs["iteration"], self._data.size)),
                             self)
            iteration = kwargs["iteration"]
            # remove the `iteration` key from the keyword arguments so it does not get passed onto the solution data
            # storage creation
            del kwargs["iteration"]
        else:
            iteration = -1

        _old_data = self._data
        if iteration == -1:
            self._data = np.append(self._data, self._data_type(*args, **kwargs))
        else:
            self._data = np.insert(self._data, iteration, self._data_type(*args, **kwargs))

        try:
            self._check_consistency()
        except ValueError:
            # consistency check failed, thus removing recently added solution data storage
            warnings.warn("Consistency Check failed. Not adding this solution.")
            self._data = _old_data.copy()
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
            `-1` means last iteration.

        Returns
        -------
        solution : :py:class:`.solutions.ISolutionData`
            or :py:class:`None` if no solutions are stored.

        Raises
        ------
        ValueError :
            if given `iteration` index is not in the valid range
        """
        if self._data.size > 0:
            assert_condition(iteration in range(-1, self._data.size),
                             ValueError, "Iteration index not within valid range: {:d} not in [-1, {:d}"
                                         .format(iteration, self._data.size),
                             self)
            return self._data[iteration]
        else:
            return None

    @property
    def solutions(self):
        """
        Summary
        -------
        Read-only accessor for the stored list of solution data storages.

        Returns
        -------
        values : :py:class:`numpy.ndarray` of :py:class:`.solutions.TrajectorySolutionData` objects
        """
        return self._data

    @property
    def time_points(self):
        """
        Summary
        -------
        Proxies :py:attr:`.solutions.TrajectorySolutionData.time_points`.

        Returns
        -------
        time_points : :py:class:`numpy.ndarray`, :py:class:`None`
            :py:class:`None` is returned if no solutions have yet been stored
        """
        return self._data[0].time_points if self._data.size > 0 else None

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
        if self._data.size > 0:
            _time_points = self._data[0].time_points
            for iteration in range(1, self._data.size):
                assert_condition(np.array_equal(_time_points, self._data[iteration].time_points),
                                 ValueError, "Time points of one or more stored solution data objects do not match.",
                                 self)


__all__ = ['FullSolution']
