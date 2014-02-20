# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.utilities import assert_is_instance
from pypint.solvers.states.i_solver_state import ISolverState


class ISolverCore(object):
    """Interface for the Solver's Cores
    """

    name = 'Solver Core Interface'
    """Human readable name of the solver's core
    """

    def __init__(self):
        pass

    def run(self, state, **kwargs):
        """Apply the solver core to the current state

        Parameters
        ----------
        state : :py:class:`.ISolverState`
            Current state of the solver.
        """
        assert_is_instance(state, ISolverState,
                           "The solver's state must be given: NOT {}".format(state.__class__.__name__),
                           self)

    def compute_residual(self, state, **kwargs):
        """Computes the residual of the current state

        Parameters
        ----------
        state : :py:class:`.ISolverState`
            Current state of the solver.
        """
        assert_is_instance(state, ISolverState,
                           "The solver's state must be given: NOT {}".format(state.__class__.__name__),
                           self)

    def compute_error(self, state, **kwargs):
        """Computes the error of the current state

        Parameters
        ----------
        state : :py:class:`.ISolverState`
            Current state of the solver.
        """
        assert_is_instance(state, ISolverState,
                           "The solver's state must be given: NOT {}".format(state.__class__.__name__),
                           self)


__all__ = ['ISolverCore']
