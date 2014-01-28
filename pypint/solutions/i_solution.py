# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_solution_data import ISolutionData
from ..utilities import assert_condition


class ISolution(object):
    """
    Summary
    -------
    Generalized storage for solutions of solvers.
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        solution_data_type : :py:class:`class`
            Solution data storage type.
            Must be a subclass of :py:class:`.solutions.ISolutionData`.
            Defaults to :py:class:`.solutions.ISolutionData`.

        Raises
        ------
        ValueError :
            If `solution_data_type` is not a subclass of :py:class:`.solutions.ISolutionData`.
        """
        self._data_type = ISolutionData
        if "solution_data_type" in kwargs:
            assert_condition(issubclass(kwargs["solution_data_type"], ISolutionData),
                             ValueError, "Solution data type must be an ISolutionData class: NOT {:s}"
                                         .format(kwargs["solution_data_type"].__mro__),
                             self)
            self._data_type = kwargs["solution_data_type"]
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

    def __str__(self):
        return self.__class__.__name__ + ": {:s}".format(self._data)
