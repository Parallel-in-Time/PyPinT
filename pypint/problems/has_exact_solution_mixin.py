# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.problems.i_problem import IProblem
from pypint.utilities import assert_is_callable, assert_is_instance


class HasExactSolutionMixin(object):
    """
    Summary
    -------
    Provides exact analytical solution function for a problem.
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        exact_function : callable
            (optional)
            If given initializes the problem with the exact solution function.
        """
        self._exact_function = None
        if "exact_function" in kwargs:
            self.exact_function = kwargs["exact_function"]

    def exact(self, time, phi_of_time):
        """
        Summary
        -------
        Evaluates given exact solution function at given time and with given time-dependent data.

        Parameters
        ----------
        time : float
            Time point :math:`t`

        phi_of_time : ``numpy.ndarray``
            Time-dependent data.

        Returns
        -------
        exact solution : ``numpy.ndarray``

        Raises
        ------
        ValueError
            If ``time`` is not a ``float``.
        """
        assert_is_instance(time, float, "Time must be a float.", self)
        return self._exact_function(time, phi_of_time)

    @property
    def exact_function(self):
        """
        Summary
        -------
        Accessor for the exact solution function.

        Raises
        ------
        ValueError
            On setting, if the new exact solution function is not callable.
        """
        return self._exact_function

    @exact_function.setter
    def exact_function(self, exact_function):
        assert_is_callable(exact_function, "Exact function must be callable.", self)
        self._exact_function = exact_function


def problem_has_exact_solution(problem, checking_obj=None):
    """
    Summary
    -------
    Convenience accessor for exact solution.

    Parameters
    ----------
    problem : :py:class:`.IProblem`
        The problem to check for an exact solution function.

    checking_obj : object
        (optional)
        The object calling this function for a meaningful error message.
        For debugging purposes only.

    Returns
    -------
     : boolean
        ``True`` if exact solution was given, ``False`` otherwise

    Raises
    ------
    ValueError
        If the given problem is not an instance of :py:class:`.IProblem`.
    """
    assert_is_instance(problem, IProblem, "It needs to be a problem to have an exact solution.", checking_obj)
    return isinstance(problem, HasExactSolutionMixin)


__all__ = ['problem_has_exact_solution', 'HasExactSolutionMixin']
