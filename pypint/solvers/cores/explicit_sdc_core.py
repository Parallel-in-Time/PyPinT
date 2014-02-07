# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.kaltt@fz-juelich.de>
"""
from pypint.solvers.cores.sdc_solver_core import SdcSolverCore
from pypint.solvers.states.sdc_solver_state import SdcSolverState
from pypint.utilities import assert_is_instance, assert_is_key


class ExplicitSdcCore(SdcSolverCore):
    name = "Explicit SDC"

    def __init__(self):
        super(ExplicitSdcCore, self).__init__()

    def run(self, state, **kwargs):
        """
        Summary
        -------
        Explicit Euler step method.

        Extended Summary
        ----------------
        .. math::

            u_{m+1}^{k+1} = u_m^{k+1} + \\Delta_\\tau \\left( F(t_m, u_m^{k+1}) - F(t_m, u_m^k) \\right)
                                      + \\Delta_t I_m^{m+1} \\left( F(\\vec{u}^k) \\right)

        Parameters
        ----------
        solver state : :py:class:`.SdcSolverState`
        """
        super(ExplicitSdcCore, self).run(state, **kwargs)

        assert_is_instance(state, SdcSolverState,
                           "State must be an SdcSolverState: NOT {:s}".format(state.__class__.__name__),
                           self)

        assert_is_key(kwargs, 'problem',
                      "The problem is required as a proxy to the implicit space solver.",
                      self)
        _problem = kwargs['problem']

        # using step-wise formula
        # Formula:
        #   u_{m+1}^{k+1} = u_m^{k+1} + \Delta_\tau [ F(u_m^{k+1}) - F(u_m^k) ] + \Delta_t I_m^{m+1}(F(u^k))
        _previous_step_index = state.previous_step_index
        _current_time_step_index = state.current_time_step_index
        state.current_step.solution = \
            (state.current_time_step.previous_step.solution
             + state.current_step.delta_tau
             * (_problem.evaluate(state.current_step.time_point,
                                  state.current_time_step.previous_step.solution)
                - _problem.evaluate(state.current_step.time_point,
                                    state.previous_iteration[_current_time_step_index][_previous_step_index].solution))
             + state.delta_interval * state.current_step.integral)


__all__ = ['ExplicitSdcCore']
