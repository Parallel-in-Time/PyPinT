# coding=utf-8

from ...problems import problem_has_direct_implicit
from ...utilities.tracing import assert_is_key
import numpy as np


class SdcCoreMixin(object):
    class CoreState(object):
        _num_nodes = 0

        def __init__(self):
            self.time_step_index = 0       # t
            self.current_point_index = 0   # _i
            self.node_index = 0            # n
            self.first_node_index = 0      # _i0
            self.last_node_index = 0       # _i1
            self.delta_tau = 0.0           # _dt
            self.current_time_point = 0.0  # _t0
            self.next_time_point = 0.0     # _t1

        def calculate_current_point_index(self):
            self.current_point_index = self.time_step_index * (SdcCoreMixin.CoreState._num_nodes - 1) + self.node_index

        def calculate_node_range(self):
            self.first_node_index = self.current_point_index - self.node_index
            self.last_node_index = (self.time_step_index + 1) * (SdcCoreMixin.CoreState._num_nodes - 1)

        @property
        def previous_point_index(self):
            return self.current_point_index - 1

    def __init__(self):
        self.CoreState._num_nodes = self.num_nodes
        self._core_state = SdcCoreMixin.CoreState()

    def execute_core(self, **kwargs):
        if self.is_implicit:
            self.implicit_euler(**kwargs)

        elif self.is_semi_implicit:
            self.semi_implicit_euler(**kwargs)

        elif self.is_explicit:
            self.explicit_euler(**kwargs)

        else:
            # should not reach here
            pass

    def explicit_euler(self, **kwargs):
        assert_is_key(kwargs, "integral", "Explicit Euler needs the precalculated integral.")
        # using step-wise formula
        # Formula:
        #   u_{m+1}^{k+1} = u_m^{k+1} + \Delta_\tau [ F(u_m^{k+1}) - F(u_m^k) ] + \Delta_t I_m^{m+1}(F(u^k))
        self.current_state\
            .solution_at(self.core_state.current_point_index,
                         (self.current_state.solution[self.core_state.previous_point_index]
                          + self.core_state.delta_tau
                          * (self.problem.evaluate(self.core_state.current_time_point,
                                                   self.current_state.solution[self.core_state.previous_point_index])
                             - self.problem.evaluate(self.core_state.current_time_point,
                                                     self.previous_state.solution[self.core_state.previous_point_index])
                             )
                          + self._deltas["I"] * kwargs['integral']))

    def implicit_euler(self, **kwargs):
        assert_is_key(kwargs, "integral", "Implicit Euler needs the precalculated integral.")
        if problem_has_direct_implicit(self.problem, self):
            _sol = \
                self.problem\
                    .direct_implicit(phis_of_time=[self.previous_state.solution[self.core_state.previous_point_index],
                                                   self.previous_state.solution[self.core_state.current_point_index],
                                                   self.current_state.solution[self.core_state.previous_point_index]],
                                     delta_node=self.core_state.delta_tau,
                                     delta_step=self._deltas["I"],
                                     integral=kwargs['integral'])
        else:
            # using step-wise formula
            #   u_{m+1}^{k+1} - \Delta_\tau F(u_{m+1}^{k+1})
            #     = u_m^{k+1} - \Delta_\tau F(u_m^k) + \Delta_t I_m^{m+1}(F(u^k))
            _expl_term = \
                self.current_state.solution[self.core_state.previous_point_index] \
                - self.core_state.delta_tau \
                * self.problem.evaluate(self.core_state.next_time_point,
                                        self.previous_state.solution[self.core_state.current_point_index]) \
                + self._deltas["I"] * kwargs['integral']
            _func = \
                lambda x_next: \
                    _expl_term \
                    + self.core_state.delta_tau * self.problem.evaluate(self.core_state.next_time_point, x_next) \
                    - x_next
            _sol = \
                self.problem.implicit_solve(np.array([self.current_state.solution[self.core_state.current_point_index]],
                                                     dtype=self.problem.numeric_type), _func)
        self.current_state.solution_at(self.core_state.current_point_index,
                                       _sol if type(self.current_state.solution[self.core_state.current_point_index]) == type(_sol) else _sol[0])

    def semi_implicit_euler(self, **kwargs):
        assert_is_key(kwargs, "integral", "Semi-Implicit Euler needs the precalculated integral.")
        if problem_has_direct_implicit(self.problem, self):
            _sol = \
                self.problem\
                    .direct_implicit(phis_of_time=[self.previous_state.solution[self.core_state.previous_point_index],
                                                   self.previous_state.solution[self.core_state.current_point_index],
                                                   self.current_state.solution[self.core_state.previous_point_index]],
                                     delta_node=self.core_state.delta_tau,
                                     delta_step=self._deltas["I"],
                                     integral=kwargs['integral'])
        else:
            _expl_term = \
                self.current_state.solution[self.core_state.previous_point_index] \
                + self.core_state.delta_tau \
                * (self.problem.evaluate(self.core_state.current_time_point,
                                         self.current_state.solution[self.core_state.previous_point_index],
                                         partial="expl")
                   - self.problem.evaluate(self.core_state.current_time_point,
                                           self.previous_state.solution[self.core_state.previous_point_index],
                                           partial="expl")
                   - self.problem.evaluate(self.core_state.next_time_point,
                                           self.previous_state.solution[self.core_state.current_point_index],
                                           partial="impl")) \
                + self._deltas["I"] * kwargs['integral']
            _func = \
                lambda x_next: \
                    _expl_term \
                    + self.core_state.delta_tau * self.problem.evaluate(self.core_state.next_time_point,
                                                                        x_next, partial="impl") \
                    - x_next
            _sol = \
                self.problem.implicit_solve(np.array([self.current_state.solution[self.core_state.current_point_index]],
                                                     dtype=self.problem.numeric_type), _func)
        self.current_state.solution_at(self.core_state.current_point_index,
                                       _sol if type(self.current_state.solution[self.core_state.current_point_index]) == type(_sol) else _sol[0])

    @property
    def core_state(self):
        return self._core_state
