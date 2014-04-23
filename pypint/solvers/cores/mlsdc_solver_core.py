# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.solvers.cores.i_solver_core import ISolverCore
from pypint.solvers.states.mlsdc_solver_state import MlSdcStepState
from pypint.problems.has_exact_solution_mixin import problem_has_exact_solution
from pypint.problems import IProblem
from pypint.solvers.diagnosis import Error, Residual
from pypint.utilities import assert_named_argument, assert_is_instance


class MlSdcSolverCore(ISolverCore):
    """Provides the Step-Method-Core for :py:class:`.MlSdc` solver.

    This is to be used as a Mixin for the :py:class:`.MlSdc` solver to provide the core step-methods such as the
    explicit, implicit and semi-implicit Euler.

    Notes
    -----
    The scope of `self` must be seen in the context of a :py:class:`.MlSdc` solver instance here.
    Thus, access to :py:attr:`.MlSdc.problem` or :py:attr:`.MlSdc.is_implicit` is perfectly fine (though IDEs will not
    resolve this correctly).

    As well, note, that :py:meth:`.SdcCoreMixin.__init__` must be called explicitly and is not called via
    :py:func:`super` calls.
    :py:meth:`.SdcCoreMixin.__init__` is called by :py:meth:`.Sdc.init`.
    """

    name = 'MLSDC Solver Core'

    def __init__(self):
        super(MlSdcSolverCore, self).__init__()

    def run(self, state, **kwargs):
        super(MlSdcSolverCore, self).run(state, **kwargs)

    def compute_residual(self, state, **kwargs):
        # LOG.debug("computing residual")
        super(MlSdcSolverCore, self).compute_residual(state, **kwargs)
        _step = kwargs['step'] if 'step' in kwargs else state.current_step
        if _step.fas_correction is None:
            _step.solution.residual = Residual(
                abs(state.current_level.initial.value
                    + state.delta_interval * kwargs['integral']
                    - _step.value)
            )
        else:
            _step.solution.residual = Residual(
                abs(state.current_level.initial.value
                    + state.delta_interval * kwargs['integral']
                    - _step.value + _step.fas_correction)
            )
        # LOG.debug("Residual: {: .4f} = | {: .4f} + {: .4f} * {: .4f} - {: .4f} |"
        #           .format(state.current_step.solution.residual.value[0],
        #                   state.current_time_step.initial.value[0],
        #                   state.delta_interval, kwargs['integral'][0],
        #                   state.current_step.value[0]))

    def compute_error(self, state, **kwargs):
        super(MlSdcSolverCore, self).compute_error(state, **kwargs)

        assert_named_argument('problem', kwargs, types=IProblem, descriptor="Problem", checking_obj=self)

        _problem = kwargs['problem']

        if problem_has_exact_solution(_problem, self):
            # LOG.debug("Error for t={:.3f}: {} - {}".format(state.current_step.time_point,
            #                                               state.current_step.value,
            #                                               _problem.exact(state.current_step.time_point)))
            state.current_step.solution.error = Error(
                abs(state.current_step.value - _problem.exact(state.current_step.time_point))
            )
        else:
            # we need the exact solution for that
            #  (unless we find an error approximation method)
            pass

    def _previous_iteration_previous_step(self, state):
        if state.previous_iteration_index is not None:
            if state.previous_step_index is not None:
                assert_is_instance(state.previous_iteration[state.current_level_index][state.previous_step_index],
                                   MlSdcStepState)
                return state.previous_iteration[state.current_level_index][state.previous_step_index]
            else:
                assert_is_instance(state.previous_iteration[state.current_level_index].initial,
                                   MlSdcStepState)
                return state.previous_iteration[state.current_level_index].initial
        else:
            assert_is_instance(state.initial,
                               MlSdcStepState)
            return state.initial

    def _previous_iteration_current_step(self, state):
        if state.previous_iteration_index is not None:
            return state.previous_iteration[state.current_level_index][state.current_step_index]
        else:
            return state.initial


__all__ = ['MlSdcSolverCore']
