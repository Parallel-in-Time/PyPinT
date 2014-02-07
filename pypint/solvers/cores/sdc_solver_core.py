# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.solvers.cores.i_solver_core import ISolverCore
from pypint.problems.has_exact_solution_mixin import problem_has_exact_solution
from pypint.solvers.diagnosis import Error, Residual
from pypint.utilities import assert_is_key


class SdcSolverCore(ISolverCore):
    """
    Summary
    -------
    Provides the Step-Method-Core for :py:class:`.Sdc` solver.

    Extended Summary
    ----------------
    This is to be used as a Mixin for the :py:class:`.Sdc` solver to provide the core step-methods such as the explicit,
    implicit and semi-implicit Euler.

    Notes
    -----
    The scope of `self` must be seen in the context of a :py:class:`.Sdc` solver instance here.
    Thus, access to :py:attr:`.Sdc.problem` or :py:attr:`.Sdc.is_implicit` is perfectly fine (though IDEs will not
    resolve this correctly).

    As well, note, that :py:meth:`.SdcCoreMixin.__init__` must be called explicitly and is not called via
    :py:func:`super` calls.
    :py:meth:`.SdcCoreMixin.__init__` is called by :py:meth:`.Sdc.init`.
    """

    def __init__(self):
        super(SdcSolverCore, self).__init__()

    def run(self, state, **kwargs):
        super(SdcSolverCore, self).run(state, **kwargs)

    def compute_residual(self, state, **kwargs):
        super(SdcSolverCore, self).compute_residual(state, **kwargs)
        state.current_step.solution.residual = Residual(
            abs(state.current_time_step[0].solution.value
                + state.delta_interval * state.current_step.integral
                - state.current_step.solution.value)
        )

    def compute_error(self, state, **kwargs):
        super(SdcSolverCore, self).compute_error(state, **kwargs)

        assert_is_key(kwargs, 'problem',
                      "The problem is required as a proxy to the implicit space solver.",
                      self)
        _problem = kwargs['problem']

        if problem_has_exact_solution(self.problem, self):
            state.current_step.solution.error = Error(
                abs(state.current_step.solution.value - _problem.exact(state.current_step.time_point))
            )
        else:
            # we need the exact solution for that
            #  (unless we find an error approximation method)
            pass


__all__ = ['SdcSolverCore']
