# coding=utf-8

from ...problems import problem_has_direct_implicit
from ...utilities.tracing import assert_is_key
import numpy as np


class SdcCoreMixin(object):
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

    class CoreState(object):
        """
        Summary
        -------
        Internal state of :py:class:`.Sdc` step-method core providing
        """
        _num_nodes = 0

        def __init__(self):
            """
            Attributes
            ----------
            time_step_ndx : int
                This is :math:`t`
            curr_pnt_ndx : int
            node_ndx : int
                This is :math:`m`
            first_node_ndx : int
                This is :math:`\\tau_0`
            last_node_ndx : int
                This is :math:`\\tau_M`
            delta_tau : float
                This is :math:`\\Delta_tau`
            curr_time_pnt : float
            next_time_pnt : float
            """
            self.time_step_ndx = 0    # t
            self.curr_pnt_ndx = 0     # _i
            self.node_ndx = 0         # n
            self.first_node_ndx = 0   # _i0
            self.last_node_ndx = 0    # _i1
            self.delta_tau = 0.0      # _dt
            self.curr_time_pnt = 0.0  # _t0
            self.next_time_pnt = 0.0  # _t1

        def calculate_current_point_index(self):
            self.curr_pnt_ndx = self.time_step_ndx * (SdcCoreMixin.CoreState._num_nodes - 1) + self.node_ndx

        def calculate_node_range(self):
            self.first_node_ndx = self.curr_pnt_ndx - self.node_ndx
            self.last_node_ndx = (self.time_step_ndx + 1) * (SdcCoreMixin.CoreState._num_nodes - 1)

        @property
        def prev_pnt_ndx(self):
            return self.curr_pnt_ndx - 1

    def __init__(self):
        self.CoreState._num_nodes = self.num_nodes
        self._core_state = SdcCoreMixin.CoreState()

    def execute_core(self, **kwargs):
        """
        Summary
        -------
        Executes the core step-method based on current setup of the :py:class:`.Sdc` solver.

        Extended Summary
        ----------------
        Currently behaves like:

        :py:attr:`.Sdc.is_implicit` is True :
            executing implicit Euler (see :py:meth:`.implicit_euler`)
        :py:attr:`.Sdc.is_explicit` is True :
            executing explicit Euler (see :py:meth:`.explicit_euler`)
        :py:attr:`.Sdc.is_semi_implicit` is True :
            executing semi-implicit Euler (see :py:meth:`.semi_implicit_euler`)

        See the actual step methods for expected parameters.
        """
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
        integral : float
            Precalculated integral :math:`I_m^{m+1}` of the right hand side function.
        """
        assert_is_key(kwargs, "integral", "Explicit Euler needs the precalculated integral.")
        # using step-wise formula
        # Formula:
        #   u_{m+1}^{k+1} = u_m^{k+1} + \Delta_\tau [ F(u_m^{k+1}) - F(u_m^k) ] + \Delta_t I_m^{m+1}(F(u^k))
        self.current_state\
            .solution_at(self.core_state.curr_pnt_ndx,
                         (self.current_state.solution[self.core_state.prev_pnt_ndx]
                          + self.core_state.delta_tau
                          * (self.problem.evaluate(self.core_state.curr_time_pnt,
                                                   self.current_state.solution[self.core_state.prev_pnt_ndx])
                             - self.problem.evaluate(self.core_state.curr_time_pnt,
                                                     self.previous_state.solution[self.core_state.prev_pnt_ndx])
                             )
                          + self._deltas["I"] * kwargs['integral']))

    def implicit_euler(self, **kwargs):
        """
        Summary
        -------
        Implicit Euler step method.

        Extended Summary
        ----------------
        .. math::

            u_{m+1}^{k+1} - \\Delta_\\tau F(t_{m+1}, u_{m+1}^{k+1}) =
                u_m^{k+1} + \\Delta_\\tau F(t_{m+1}, u_{m+1}^k) + \\Delta_t I_m^{m+1} \\left( F(\\vec{u}^k) \\right)

        Parameters
        ----------
        integral : float
            Precalculated integral :math:`I_m^{m+1}` of the right hand side function.
        """
        assert_is_key(kwargs, "integral", "Implicit Euler needs the precalculated integral.")
        if problem_has_direct_implicit(self.problem, self):
            _sol = \
                self.problem\
                    .direct_implicit(phis_of_time=[self.previous_state.solution[self.core_state.prev_pnt_ndx],
                                                   self.previous_state.solution[self.core_state.curr_pnt_ndx],
                                                   self.current_state.solution[self.core_state.prev_pnt_ndx]],
                                     delta_node=self.core_state.delta_tau,
                                     delta_step=self._deltas["I"],
                                     integral=kwargs['integral'])
        else:
            # using step-wise formula
            #   u_{m+1}^{k+1} - \Delta_\tau F(u_{m+1}^{k+1})
            #     = u_m^{k+1} - \Delta_\tau F(u_m^k) + \Delta_t I_m^{m+1}(F(u^k))
            _expl_term = \
                self.current_state.solution[self.core_state.prev_pnt_ndx] \
                - self.core_state.delta_tau \
                * self.problem.evaluate(self.core_state.next_time_pnt,
                                        self.previous_state.solution[self.core_state.curr_pnt_ndx]) \
                + self._deltas["I"] * kwargs['integral']
            _func = \
                lambda x_next: \
                    _expl_term \
                    + self.core_state.delta_tau * self.problem.evaluate(self.core_state.next_time_pnt, x_next) \
                    - x_next
            _sol = \
                self.problem.implicit_solve(np.array([self.current_state.solution[self.core_state.curr_pnt_ndx]],
                                                     dtype=self.problem.numeric_type), _func)
        self.current_state.solution_at(self.core_state.curr_pnt_ndx,
                                       _sol if type(self.current_state.solution[self.core_state.curr_pnt_ndx]) == type(_sol) else _sol[0])

    def semi_implicit_euler(self, **kwargs):
        """
        Summary
        -------
        Semi-Implicit Euler step method.

        Extended Summary
        ----------------
        .. math::

            u_{m+1}^{k+1} - \\Delta_\\tau F_I(t_{m+1}, u_{m+1}^{k+1}) =
                u_m^{k+1} + \\Delta_\\tau \\left( F_I(t_{m+1}, u_{m+1}^k)
                                                  - F_E(t_m, u_m^{k+1}) + F_E(t_m, u_m^k) \\right)
                          + \\Delta_t I_m^{m+1} \\left( F(\\vec{u}^k) \\right)

        Parameters
        ----------
        integral : float
            Precalculated integral :math:`I_m^{m+1}` of the right hand side function.

        Notes
        -----
        This step method requires the given problem to provide partial evaluation of the right-hand side.
        """
        assert_is_key(kwargs, "integral", "Semi-Implicit Euler needs the precalculated integral.")
        if problem_has_direct_implicit(self.problem, self):
            _sol = \
                self.problem\
                    .direct_implicit(phis_of_time=[self.previous_state.solution[self.core_state.prev_pnt_ndx],
                                                   self.previous_state.solution[self.core_state.curr_pnt_ndx],
                                                   self.current_state.solution[self.core_state.prev_pnt_ndx]],
                                     delta_node=self.core_state.delta_tau,
                                     delta_step=self._deltas["I"],
                                     integral=kwargs['integral'])
        else:
            _expl_term = \
                self.current_state.solution[self.core_state.prev_pnt_ndx] \
                + self.core_state.delta_tau \
                * (self.problem.evaluate(self.core_state.curr_time_pnt,
                                         self.current_state.solution[self.core_state.prev_pnt_ndx],
                                         partial="expl")
                   - self.problem.evaluate(self.core_state.curr_time_pnt,
                                           self.previous_state.solution[self.core_state.prev_pnt_ndx],
                                           partial="expl")
                   - self.problem.evaluate(self.core_state.next_time_pnt,
                                           self.previous_state.solution[self.core_state.curr_pnt_ndx],
                                           partial="impl")) \
                + self._deltas["I"] * kwargs['integral']
            _func = \
                lambda x_next: \
                    _expl_term \
                    + self.core_state.delta_tau * self.problem.evaluate(self.core_state.next_time_pnt,
                                                                        x_next, partial="impl") \
                    - x_next
            _sol = \
                self.problem.implicit_solve(np.array([self.current_state.solution[self.core_state.curr_pnt_ndx]],
                                                     dtype=self.problem.numeric_type), _func)
        self.current_state.solution_at(self.core_state.curr_pnt_ndx,
                                       _sol if type(self.current_state.solution[self.core_state.curr_pnt_ndx]) == type(_sol) else _sol[0])

    @property
    def core_state(self):
        """
        Summary
        -------
        Accessor to the internal core state.
        """
        return self._core_state
