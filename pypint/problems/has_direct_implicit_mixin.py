# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.problems.i_problem import IProblem
from pypint.utilities import assert_is_instance


class HasDirectImplicitMixin(object):
    """Provides direct implicit formulation of the problem.
    """
    def __init__(self, *args, **kwargs):
        pass

    def direct_implicit(self, *args, **kwargs):
        """
        Raises
        ------
        NotImplementedError :
            If the problem using this Mixin actually does not override this method.
        """
        raise NotImplementedError("If this mixin is used, the problem must implement this function.")


def problem_has_direct_implicit(problem, checking_obj=None):
    """Convenience checker for existence of a direct implicit formulation of a problem.

    Parameters
    ----------
    problem : :py:class:`.IProblem`
        The problem to check for a direct implicit formulation.
    checking_obj : :py:class:`object`
        *(optional)*
        The object calling this function for a meaningful error message.
        For debugging purposes only.

    Returns
    -------
    has_direct_impl : :py:class:`bool`
        :py:class:`True` if exact solution was given, :py:class:`False` otherwise

    Raises
    ------
    ValueError :
        If the given problem is not an instance of :py:class:`.IProblem`.
    """
    assert_is_instance(problem, IProblem,
                       message="It needs to be a problem to have a direct implicit formula.", checking_obj=checking_obj)
    return isinstance(problem, HasDirectImplicitMixin)


__all__ = ['problem_has_direct_implicit', 'HasDirectImplicitMixin']
