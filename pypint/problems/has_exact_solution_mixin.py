# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.problems.i_problem import IProblem
from pypint.utilities import assert_is_callable, assert_is_instance


class HasExactSolutionMixin(object):
    """Provides exact analytical solution function for a problem.
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        exact_function : :py:class:`callable`
            *(optional)*
            If given initializes the problem with the exact solution function.
        """
        self._exact_function = None
        if "exact_function" in kwargs:
            self.exact_function = kwargs["exact_function"]

    def exact(self, time):
        """Evaluates given exact solution function at given time and with given time-dependent data.

        Parameters
        ----------
        time : :py:class:`float`
            Time point :math:`t`

        Returns
        -------
        exact_solution : :py:class:`numpy.ndarray`

        Raises
        ------
        ValueError :

            * if ``time`` is not a :py:class:`float`
            * if ``phi_of_time`` is not a :py:class:`numpy.ndarray`
            * if not exact function is given
        """
        assert_is_instance(time, float, descriptor="Time Point", checking_obj=self)
        assert_is_callable(self._exact_function, descriptor="Exact Function", checking_obj=self)
        return self._exact_function(time)

    @property
    def exact_function(self):
        """Accessor for the exact solution function.

        Raises
        ------
        ValueError :
            On setting, if the new exact solution function is not callable.
        """
        return self._exact_function

    @exact_function.setter
    def exact_function(self, exact_function):
        assert_is_callable(exact_function, descriptor="Exact Function", checking_obj=self)
        self._exact_function = exact_function


def problem_has_exact_solution(problem, checking_obj=None):
    """Convenience accessor for exact solution.

    Parameters
    ----------
    problem : :py:class:`.IProblem`
        The problem to check for an exact solution function.
    checking_obj : object
        *(optional)*
        The object calling this function for a meaningful error message.
        For debugging purposes only.

    Returns
    -------
    has_exact_solution : :py:class:`bool`
        :py:class:`True` if exact solution was given, :py:class:`False` otherwise

    Raises
    ------
    ValueError :
        If the given problem is not an instance of :py:class:`.IProblem`.
    """
    assert_is_instance(problem, IProblem, message="It needs to be a problem to have an exact solution.",
                       checking_obj=checking_obj)
    return isinstance(problem, HasExactSolutionMixin)


__all__ = ['problem_has_exact_solution', 'HasExactSolutionMixin']
