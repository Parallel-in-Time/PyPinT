# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.kaltt@fz-juelich.de>
"""
import numpy as np

from pypint.solvers.cores.mlsdc_solver_core import MlSdcSolverCore
from pypint.solvers.states.mlsdc_solver_state import MlSdcSolverState
from pypint.problems import IProblem
from pypint.problems.has_direct_implicit_mixin import problem_has_direct_implicit
from pypint.utilities import assert_is_instance, assert_named_argument
from pypint.utilities.logging import LOG


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

        use_intermediate = kwargs['use_intermediate'] if 'use_intermediate' in kwargs else False

        if use_intermediate:
            # LOG.debug("using intermediate")
            _previous_iteration_current_step = state.current_iteration.current_level.current_step.intermediate
        elif not state.current_iteration.on_finest_level:
            _previous_iteration_current_step = state.current_iteration.current_level.current_step
        else:
            _previous_iteration_current_step = self._previous_iteration_current_step(state)
        if not _previous_iteration_current_step.rhs_evaluated:
            _previous_iteration_current_step.rhs = \
                _problem.evaluate_wrt_time(_previous_iteration_current_step.time_point,
                                           _previous_iteration_current_step.value)

        if not state.current_iteration.on_finest_level:
            _previous_iteration_previous_step = state.current_iteration.current_level.previous_step
        else:
            _previous_iteration_previous_step = self._previous_iteration_previous_step(state)
        if not _previous_iteration_previous_step.rhs_evaluated:
            _previous_iteration_previous_step.rhs = \
                _problem.evaluate_wrt_time(_previous_iteration_previous_step.time_point,
                                           _previous_iteration_previous_step.value)

        _fas = np.zeros(_previous_iteration_current_step.rhs.shape,
                        dtype=_previous_iteration_current_step.rhs.dtype)
        if not use_intermediate and _previous_iteration_current_step.has_fas_correction():
            # LOG.debug("   previous iteration current step has FAS: %s"
            #           % _previous_iteration_current_step.fas_correction)
            _fas = _previous_iteration_current_step.fas_correction

        if problem_has_direct_implicit(_problem, self):
            if not state.current_iteration.on_finest_level:
                _previous_iteration_previous_step = state.current_iteration.current_level.previous_step
            else:
                _previous_iteration_previous_step = self._previous_iteration_previous_step(state)

            _sol = _problem.direct_implicit(phis_of_time=[_previous_iteration_previous_step.value,
                                                          _previous_iteration_current_step.value,
                                                          state.previous_step.value],
                                            delta_node=state.current_step.delta_tau,
                                            integral=state.current_step.integral,
                                            fas=_fas,
                                            core=self)
        else:
            # using step-wise formula
            #   u_{m+1}^{k+1} - \Delta_\tau F(u_{m+1}^{k+1})
            #     = u_m^{k+1} - \Delta_\tau F(u_m^k) + \Delta_t I_m^{m+1}(F(u^k))
            # Note: \Delta_t is always 1.0 as it's part of the integral
            _expl_term = \
                state.current_time_step.previous_step.value \
                - state.current_step.delta_tau * _previous_iteration_current_step.rhs \
                + state.current_step.integral + _fas
            _func = lambda x_next: \
                _expl_term \
                + state.current_step.delta_tau * _problem.evaluate_wrt_time(state.current_step.time_point, x_next) \
                - x_next
            _sol = _problem.implicit_solve(state.current_step.value, _func)

        if type(state.current_step.value) == type(_sol):
            state.current_step.value = _sol
        else:
            state.current_step.value = _sol[0]


__all__ = ['ImplicitMlSdcCore']
