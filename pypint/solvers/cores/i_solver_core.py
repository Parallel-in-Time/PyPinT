# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.utilities import assert_is_instance
from pypint.solvers.states.i_solver_state import ISolverState


class ISolverCore(object):
    def __init__(self):
        pass

    def run(self, state, **kwargs):
        assert_is_instance(state, ISolverState,
                           "",
                           self)

    def compute_residual(self, state, **kwargs):
        assert_is_instance(state, ISolverState,
                           "",
                           self)

    def compute_error(self, state, **kwargs):
        assert_is_instance(state, ISolverState,
                           "",
                           self)


__all__ = ['ISolverCore']
