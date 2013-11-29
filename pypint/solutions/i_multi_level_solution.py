# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.solutions.i_solution import ISolution
import numpy as np
from pypint.utilities import func_name


class IMultiLevelSolution(ISolution):
    """
    Summary
    -------
    Storage for solutions of multi-level iterative solvers.

    .. todo:: Adjust and fix handling of number of iterations per level.
    """
    def __init__(self):
        super(IMultiLevelSolution, self).__init__()
        self._values = np.zeros(0, dtype=np.ndarray)
        self._used_levels = None

    def add_solution(self, points, values, *args, **kwargs):
        """
        Summary
        -------
        Adds a new solution of the specified iteration.

        Parameters
        ----------
        data : numpy.ndarray
             solution data

        kwargs : dict

            ``level`` : integer
                Index of the level of this solution (0-based).
                ``-1`` adds a new level to the end.

            ``iteration`` : integer
                Index of the iteration of the level's solver (1-based).
                ``-1`` adds a new iteration to the end.

        Raises
        ------
        ValueError
            * if either ``level`` or ``iteration`` are not given
            * if there are more than ``iteration`` solutions already stored for the given ``level``

        See Also
        --------
        .ISolution.add_solution
            overridden method
        """
        super(IMultiLevelSolution, self).add_solution(points, values, args, kwargs)
        if "level" not in kwargs:
            kwargs["level"] = -1

        level = int(kwargs["level"])
        _old_level_size = self._values.size

        # resize data to fit specified level
        if level == -1 or level >= _old_level_size:
            if level == -1:
                _level_resize = _old_level_size + 1
            else:
                _level_resize = level + 1
            # create new index at the end of the data
            self._values = np.resize(self._values, _level_resize)
            self._values[level] = np.zeros(0, dtype=np.ndarray)

        if "iteration" not in kwargs:
            kwargs["iteration"] = -1
        iteration = int(kwargs["iteration"])
        _old_iter_size = self._values[level].size
        # get True for each empty iteration
        _empty_iter_mask = np.ma.masked_equal(self._values[level], None).mask

        # resize data of level to fit specified iteration
        if iteration == -1 or iteration >= _old_iter_size:
            if iteration == -1:
                _iter_resize = _old_iter_size + 1
            else:
                _iter_resize = iteration + 1
            # create new index at the end of the data
            self._values[level] = np.resize(self._values[level], _iter_resize)
            # and set newly created value to None
            self._values[level][iteration] = None

        if iteration != -1 and self._values[level][iteration] is not None:
            raise ValueError(func_name(self) +
                             "Data for iteration {:d} of level {:d} is already present. "
                             .format(iteration, level) + " Not overriding.")

        # fill in non-set iterations
        _empty_iter_mask = np.concatenate((_empty_iter_mask,
                                           [True] * (self._values[level].size - _old_iter_size)))
        self._values[level][_empty_iter_mask] = None

        self._values[level][iteration] = np.array(values, dtype=np.float64)
        # TODO: handle iteration count

    def solution(self, *args, **kwargs):
        """
        Summary
        -------
        Queries the solution vector of the given iteration.

        Parameters
        ----------
        kwargs : dict

            ``level`` : integer
                Index of the desired level of the solution vector.
                Defaults to -1.
                Index ``0`` is the first, finest, level, ``-1`` the last, coarsest, level.

            ``iteration`` : integer
                Index of the desired solution vector.
                Defaults to -1.
                Index ``1`` is the first solution, ``-1`` the last and final solution vector.

        Returns
        -------
        solution vector : numpy.ndarray
            Solution of the given iteration and level.

        Raises
        ------
        ValueError
            * if ``level`` is not available.
            * if ``iteration`` is not available in specified level.

        See Also
        --------
        .ISolution.solution
            overridden method
        """
        super(IMultiLevelSolution, self).solution(args, kwargs)
        if "level" not in kwargs:
            level = -1
        else:
            level = kwargs["level"]

        if level != -1 and self._values.size > level:
            raise ValueError(func_name(self) +
                             "Desired level is not available: {:d}"
                             .format(level))

        if "iteration" not in kwargs:
            iteration = -1
        else:
            iteration = kwargs["iteration"]

        if iteration != -1 and iteration > self._values[level].size:
            raise ValueError(func_name(self) +
                             "Desired iteration is not available in levle {:d}: {:d}"
                             .format(level, iteration))

        return self._values[level][iteration]

    @property
    def used_levels(self):
        """
        Summary
        -------
        Accessor for the number of levels.

        Parameters
        ----------
        used_levels : integer
            Number of used levels to be set.

        Returns
        -------
        used levels : integer
            Number of used levels.
        """
        return self._used_levels

    @used_levels.setter
    def used_levels(self, used_levels):
        self._used_levels = int(used_levels)
