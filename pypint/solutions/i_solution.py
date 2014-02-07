# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from copy import deepcopy

from pypint.utilities import assert_condition


class ISolution(object):
    """
    Summary
    -------
    Generalized storage for solutions of solvers.
    """

    def __init__(self, *args, **kwargs):
        self._data_type = None
        self._data = None
        self._used_iterations = 0
        self._reduction = None

    def add_solution(self, *args, **kwargs):
        """
        Summary
        -------
        Adds a new solution data storage object.

        Raises
        ------
        NotImplementedError :
            If called directly or via :py:meth:`super`.

        Notes
        -----
        This method must be overridden in derived classes.
        """
        raise NotImplementedError("Must be implemented and overridden by subclasses.")

    @property
    def used_iterations(self):
        """
        Summary
        -------
        Accessor for the number of iterations.

        Parameters
        ----------
        used_iterations : :py:class:`int`
            number of used iterations

        Raises
        ------
        ValueError :
            If `used_iterations` is not a non-zero positive integer value.

        Returns
        -------
        used_iterations : :py:class:`int`
            number of used iterations
        """
        return self._used_iterations

    @used_iterations.setter
    def used_iterations(self, used_iterations):
        assert_condition(used_iterations > 0,
                         ValueError, "Number of used iterations must be non-zero positive: NOT {:d}"
                                     .format(used_iterations),
                         self)
        self._used_iterations = used_iterations

    @property
    def data_storage_type(self):
        """
        Summary
        -------
        Read-only accessor for the data storage type.

        Returns
        -------
        data_storage_type : :py:class:`.TrajectorySolutionData` or :py:class:`.StepSolutionData`
            or a derived class thereof
        """
        return self._data_type

    def __copy__(self):
        copy = self.__class__.__new__(self.__class__)
        copy.__dict__.update(self.__dict__)
        return copy

    def __deepcopy__(self, memo):
        copy = self.__class__.__new__(self.__class__)
        memo[id(self)] = copy
        for item, value in self.__dict__.items():
            setattr(copy, item, deepcopy(value, memo))
        return copy

    def __str__(self):
        return self.__class__.__name__ + ": {:s}".format(self._data)


__all__ = ['ISolution']
