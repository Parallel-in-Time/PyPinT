# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.kaltt@fz-juelich.de>
"""
from pypint.solvers.cores.mlsdc_solver_core import MlSdcSolverCore
from pypint.solvers.states.mlsdc_solver_state import MlSdcSolverState
from pypint.problems import IProblem
from pypint.problems.has_direct_implicit_mixin import problem_has_direct_implicit
from pypint.utilities import assert_is_instance, assert_named_argument


class ImplicitMlSdcCore(MlSdcSolverCore):
    """Implicit MLSDC Core
    """

    name = "Implicit SDC"

    def __init__(self):
        super(ImplicitMlSdcCore, self).__init__()

    def run(self, state, **kwargs):
        """Implicit Euler step method.

        .. math::

            u_{m+1}^{k+1} - \\Delta_\\tau F(t_{m+1}, u_{m+1}^{k+1}) =
                u_m^{k+1} + \\Delta_\\tau F(t_{m+1}, u_{m+1}^k) + \\Delta_t I_m^{m+1} \\left( F(\\vec{u}^k) \\right)

        Parameters
        ----------
        solver_state : :py:class:`.MlSdcSolverState`
        """
        super(ImplicitMlSdcCore, self).run(state, **kwargs)

        assert_is_instance(state, MlSdcSolverState, descriptor="State", checking_obj=self)
        assert_named_argument('problem', kwargs, types=IProblem, descriptor="Problem", checking_obj=self)
        _problem = kwargs['problem']

        _previous_iteration_current_step = self._previous_iteration_current_step(state)

        if problem_has_direct_implicit(_problem, self):
            _previous_iteration_previous_step = self._previous_iteration_previous_step(state)

            _sol = _problem.direct_implicit(phis_of_time=[_previous_iteration_previous_step.value,
                                                          _previous_iteration_current_step.value,
                                                          state.previous_step.value],
                                            delta_node=state.current_step.delta_tau,
                                            integral=state.current_step.integral,
                                            core=self)
        else:
            if not _previous_iteration_current_step.rhs_evaluated:
                _previous_iteration_current_step.rhs = \
                    _problem.evaluate(_previous_iteration_current_step.time_point,
                                      _previous_iteration_current_step.value)

            if _previous_iteration_current_step.fas_correction:
                _previous_iteration_current_step_rhs = \
                    _previous_iteration_current_step.rhs + _previous_iteration_current_step.fas_correction
            else:
                _previous_iteration_current_step_rhs = _previous_iteration_current_step.rhs

            # using step-wise formula
            #   u_{m+1}^{k+1} - \Delta_\tau F(u_{m+1}^{k+1})
            #     = u_m^{k+1} - \Delta_\tau F(u_m^k) + \Delta_t I_m^{m+1}(F(u^k))
            # Note: \Delta_t is always 1.0 as it's part of the integral
            _expl_term = \
                state.current_time_step.previous_step.value \
                - state.current_step.delta_tau * _previous_iteration_current_step_rhs \
                + state.current_step.integral
            _func = lambda x_next: \
                _expl_term \
                + state.current_step.delta_tau * _problem.evaluate(state.current_step.time_point, x_next) \
                - x_next
            _sol = _problem.implicit_solve(state.current_step.value, _func)

        if type(state.current_step.value) == type(_sol):
            state.current_step.value = _sol
        else:
            state.current_step.value = _sol[0]


__all__ = ['ImplicitMlSdcCore']
