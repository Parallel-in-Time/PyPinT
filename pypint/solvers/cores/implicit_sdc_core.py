# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.kaltt@fz-juelich.de>
"""
from pypint.solvers.cores.sdc_solver_core import SdcSolverCore
from pypint.solvers.states.sdc_solver_state import SdcSolverState
from pypint.problems import IProblem
from pypint.problems.has_direct_implicit_mixin import problem_has_direct_implicit
from pypint.utilities import assert_is_instance, assert_named_argument


class ImplicitSdcCore(SdcSolverCore):
    """Implicit SDC Core
    """

    name = "Implicit SDC"

    def __init__(self):
        super(ImplicitSdcCore, self).__init__()

    def run(self, state, **kwargs):
        """Implicit Euler step method.

        .. math::

            u_{m+1}^{k+1} - \\Delta_\\tau F(t_{m+1}, u_{m+1}^{k+1}) =
                u_m^{k+1} + \\Delta_\\tau F(t_{m+1}, u_{m+1}^k) + \\Delta_t I_m^{m+1} \\left( F(\\vec{u}^k) \\right)

        Parameters
        ----------
        solver_state : :py:class:`.SdcSolverState`
        """
        super(ImplicitSdcCore, self).run(state, **kwargs)

        assert_is_instance(state, SdcSolverState, descriptor="State", checking_obj=self)
        assert_named_argument('problem', kwargs, types=IProblem, descriptor="Problem", checking_obj=self)
        _problem = kwargs['problem']

        _previous_iteration_current_step = self._previous_iteration_current_step(state)

        if problem_has_direct_implicit(_problem, self):
            _previous_iteration_previous_step = self._previous_iteration_previous_step(state)

            _sol = _problem.direct_implicit(phis_of_time=[_previous_iteration_previous_step.solution.value,
                                                          _previous_iteration_current_step.solution.value,
                                                          state.current_time_step.previous_step.solution.value],
                                            delta_node=state.current_step.delta_tau,
                                            integral=state.current_step.integral,
                                            core=self)
        else:
            # using step-wise formula
            #   u_{m+1}^{k+1} - \Delta_\tau F(u_{m+1}^{k+1})
            #     = u_m^{k+1} - \Delta_\tau F(u_m^k) + \Delta_t I_m^{m+1}(F(u^k))
            # Note: \Delta_t is always 1.0 as it's part of the integral
            _expl_term = \
                state.current_time_step.previous_step.solution.value \
                - state.current_step.delta_tau \
                * _problem.evaluate(state.current_step.time_point,
                                    _previous_iteration_current_step.solution.value) \
                + state.current_step.integral
            _func = lambda x_next: \
                _expl_term \
                + state.current_step.delta_tau * _problem.evaluate(state.current_step.time_point, x_next) \
                - x_next
            _sol = _problem.implicit_solve(state.current_step.solution.value, _func)

        if type(state.current_step.solution.value) == type(_sol):
            state.current_step.solution.value = _sol
        else:
            state.current_step.solution.value = _sol[0]


__all__ = ['ImplicitSdcCore']
