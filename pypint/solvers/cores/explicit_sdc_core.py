# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.kaltt@fz-juelich.de>
"""
from pypint.solvers.cores.sdc_solver_core import SdcSolverCore
from pypint.solvers.states.sdc_solver_state import SdcSolverState
from pypint.utilities import assert_is_instance, assert_is_key


class ExplicitSdcCore(SdcSolverCore):
    """Explicit SDC Core
    """

    name = "Explicit SDC"

    def __init__(self):
        super(ExplicitSdcCore, self).__init__()

    def run(self, state, **kwargs):
        """Explicit Euler step method.

        .. math::

            u_{m+1}^{k+1} = u_m^{k+1} + \\Delta_\\tau \\left( F(t_m, u_m^{k+1}) - F(t_m, u_m^k) \\right)
                                      + \\Delta_t I_m^{m+1} \\left( F(\\vec{u}^k) \\right)

        Parameters
        ----------
        solver_state : :py:class:`.SdcSolverState`
        """
        super(ExplicitSdcCore, self).run(state, **kwargs)

        assert_is_instance(state, SdcSolverState,
                           "State must be an SdcSolverState: NOT {:s}".format(state.__class__.__name__),
                           self)

        assert_is_key(kwargs, 'problem',
                      "The problem is required as a proxy to the implicit space solver.",
                      self)
        _problem = kwargs['problem']

        _previous_step_solution = state.previous_step.solution
        _previous_iteration_previous_step_solution = self._previous_iteration_previous_step(state).solution

        # using step-wise formula
        # Formula:
        #   u_{m+1}^{k+1} = u_m^{k+1} + \Delta_\tau [ F(u_m^{k+1}) - F(u_m^k) ] + \Delta_t I_m^{m+1}(F(u^k))
        # Note: \Delta_t is always 1.0 as it's part of the integral
        state.current_step.solution.value = \
            (_previous_step_solution.value + state.current_step.delta_tau
             * (_problem.evaluate(state.current_step.time_point, _previous_step_solution.value)
                - _problem.evaluate(state.current_step.time_point, _previous_iteration_previous_step_solution.value))
             + state.current_step.integral)


__all__ = ['ExplicitSdcCore']
