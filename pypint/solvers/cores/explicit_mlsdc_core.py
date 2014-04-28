# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.kaltt@fz-juelich.de>
"""
import numpy as np

from pypint.solvers.cores.mlsdc_solver_core import MlSdcSolverCore
from pypint.solvers.states.mlsdc_solver_state import MlSdcSolverState
from pypint.problems import IProblem
from pypint.utilities import assert_is_instance, assert_named_argument
from pypint.utilities.logging import LOG


class ExplicitMlSdcCore(MlSdcSolverCore):
    """Explicit MLSDC Core
    """

    name = "Explicit MLSDC"

    def __init__(self):
        super(ExplicitMlSdcCore, self).__init__()

    def run(self, state, **kwargs):
        """Explicit Euler step method.

        .. math::

            u_{m+1}^{k+1} = u_m^{k+1} + \\Delta_\\tau \\left( F(t_m, u_m^{k+1}) - F(t_m, u_m^k) \\right)
                                      + \\Delta_t I_m^{m+1} \\left( F(\\vec{u}^k) \\right)

        Parameters
        ----------
        solver_state : :py:class:`.MlSdcSolverState`
        """
        super(ExplicitMlSdcCore, self).run(state, **kwargs)

        assert_is_instance(state, MlSdcSolverState, descriptor="State", checking_obj=self)
        assert_named_argument('problem', kwargs, types=IProblem, descriptor="Problem", checking_obj=self)

        _problem = kwargs['problem']

        _previous_step = state.previous_step

        if not state.current_iteration.on_finest_level:
            # LOG.debug("   taking previous step of this iteration instead")
            _previous_iteration_previous_step = state.current_iteration.current_level.previous_step
        else:
            _previous_iteration_previous_step = self._previous_iteration_previous_step(state)

        if not _previous_step.rhs_evaluated:
            _previous_step.rhs = _problem.evaluate(state.current_step.time_point, _previous_step.value)
        if not _previous_iteration_previous_step.rhs_evaluated:
            _previous_iteration_previous_step.rhs = \
                _problem.evaluate(state.current_step.time_point, _previous_iteration_previous_step.value)

        _fas = np.zeros(_previous_step.rhs.shape, dtype=_previous_step.rhs.dtype)
        if _previous_step.has_fas_correction():
            # LOG.debug("   previous step has FAS")
            _fas = _previous_step.fas_correction

        # Note: \Delta_t is always 1.0 as it's part of the integral
        # using step-wise formula
        # Formula:
        #   u_{m+1}^{k+1} = u_m^{k+1} + \Delta_\tau [ F(u_m^{k+1}) - F(u_m^k) ] + \Delta_t I_m^{m+1}(F(u^k))
        state.current_step.value = \
            (_previous_step.value
             + state.current_step.delta_tau * (_previous_step.rhs - _previous_iteration_previous_step.rhs)
             + state.current_step.integral + _fas)
        # LOG.debug("%s = %s + %s * (%s - %s) + %s + %s" %
        #           (state.current_step.value, _previous_step.value, state.current_step.delta_tau, _previous_step.rhs,
        #            _previous_iteration_previous_step.rhs, state.current_step.integral, _fas))


__all__ = ['ExplicitMlSdcCore']
