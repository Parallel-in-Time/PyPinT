# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from copy import deepcopy
import warnings as warnings

import numpy as np

from pypint.solvers.i_iterative_time_solver import IIterativeTimeSolver
from pypint.solvers.i_parallel_solver import IParallelSolver
from pypint.communicators.message import Message
from pypint.integrators.sdc_integrator import SdcIntegrator
from pypint.integrators.node_providers.gauss_lobatto_nodes import GaussLobattoNodes
from pypint.integrators.weight_function_providers.polynomial_weight_function import PolynomialWeightFunction
from pypint.problems import IInitialValueProblem, problem_has_exact_solution
from pypint.solvers.states.sdc_solver_state import SdcSolverState
from pypint.solvers.diagnosis import IDiagnosisValue
from pypint.solvers.diagnosis.norms import supremum_norm
from pypint.plugins.timers.timer_base import TimerBase
from pypint.utilities.threshold_check import ThresholdCheck
from pypint.utilities import assert_is_instance, assert_condition, assert_is_key, func_name
from pypint import LOG

# General Notes on Implementation
# ===============================
#
# Names and Meaning of Indices
# ----------------------------
# T_max (num_time_steps) | number of time steps
# N     (num_nodes)      | number of integration nodes per time step
# t                      | index of current time step; interval: [0, T_max)
# n                      | index of current node of current time step; interval: [1, N)
#                        |  the current node is always the next node, i.e. the node we are
#                        |  calculating the value for
# i                      | index of current point in continuous array of points


class ParallelSdc(IIterativeTimeSolver, IParallelSolver):
    """*Spectral Deferred Corrections* method for solving first order ODEs.

    The *Spectral Deferred Corrections* (SDC) method is described in [Minion2003]_ (Equation 2.7)

    Default Values:

        * :py:attr:`.ThresholdCheck.max_iterations`: 5

        * :py:attr:`.ThresholdCheck.min_reduction`: 1e-7

        * :py:attr:`.num_time_steps`: 1

        * :py:attr:`.num_nodes`: 3

    Given the total number of time steps :math:`T_{max}`, number of integration nodes per time
    step :math:`N`, current time step :math:`t \\in [0,T_{max})` and the next integration node
    to consider :math:`n \\in [0, N)`.
    Let :math:`[a,b]` be the total time interval to integrate over.
    For :math:`T_{max}=3` and :math:`N=4`, this can be visualized as::

           a                                            b
           |                                            |
           |    .    .    |    .    .    |    .    .    |
        t  0    0    0    0    1    1    1    2    2    2
        n  0    1    2    3    1    2    3    1    2    3
        i  0    1    2    3    4    5    6    7    8    9

    In general, the value at :math:`a` (i.e. :math:`t=n=i=0`) is the initial value.

    See Also
    --------
    :py:class:`.IIterativeTimeSolver` :
        implemented interface
    """
    def __init__(self, **kwargs):
        super(ParallelSdc, self).__init__(**kwargs)
        IParallelSolver.__init__(self, **kwargs)
        del self._state

        self.threshold = ThresholdCheck(min_threshold=1e-7, max_threshold=10, conditions=("residual", "iterations"))
        self.timer = TimerBase()

        self._num_time_steps = 1
        self._dt = 0.0
        self._deltas = {
            't': 0.0,
            'n': np.zeros(0)
        }

        self.__nodes_type = GaussLobattoNodes
        self.__weights_type = PolynomialWeightFunction
        self.__num_nodes = 3
        self.__exact = np.zeros(0)
        self.__time_points = {
            'steps': np.zeros(0),
            'nodes': np.zeros(0)
        }

    def init(self, problem, integrator=SdcIntegrator(), **kwargs):
        """Initializes SDC solver with given problem and integrator.

        Parameters
        ----------
        num_time_steps : :py:class:`int`
            Number of time steps to be used within the time interval of the problem.

        num_nodes : :py:class:`int`
            *(otional)*
            number of nodes per time step

        nodes_type : :py:class:`.INodes`
            *(optional)*
            Type of integration nodes to be used (class name, **NOT** instance).

        weights_type : :py:class:`.IWeightFunction`
            *(optional)*
            Integration weights function to be used (class name, **NOT** instance).

        Raises
        ------
        ValueError :

            * if given problem is not an :py:class:`.IInitialValueProblem`
            * if number of nodes per time step is not given; neither through ``num_nodes``, ``nodes_type`` nor
              ``integrator``

        See Also
        --------
        :py:meth:`.IIterativeTimeSolver.init`
            overridden method (with further parameters)
        :py:meth:`.IParallelSolver.init`
            mixed in overridden method (with further parameters)
        """
        assert_is_instance(problem, IInitialValueProblem,
                           "SDC requires an initial value problem: {:s}".format(problem.__class__.__name__),
                           self)

        super(ParallelSdc, self).init(problem, integrator, **kwargs)

        if 'num_time_steps' in kwargs:
            self._num_time_steps = kwargs['num_time_steps']

        if 'num_nodes' in kwargs:
            self.__num_nodes = kwargs['num_nodes']
        elif 'nodes_type' in kwargs and kwargs['nodes_type'].num_nodes is not None:
            self.__num_nodes = kwargs['nodes_type'].num_nodes
        elif integrator.nodes_type is not None and integrator.nodes_type.num_nodes is not None:
            self.__num_nodes = integrator.nodes_type.num_nodes
        else:
            raise ValueError(func_name(self) +
                             "Number of nodes per time step not given.")

        if 'notes_type' in kwargs:
            self.__nodes_type = kwargs['notes_type']

        if 'weights_type' in kwargs:
            self.__weights_type = kwargs['weights_type']

        # TODO: need to store the exact solution somewhere else
        self.__exact = np.zeros(self.num_time_steps * (self.__num_nodes - 1) + 1, dtype=np.object)

    def run(self, core, **kwargs):
        """Applies SDC solver to the initialized problem setup.

        Solves the given problem with the explicit SDC algorithm.

        The output for the iterations explained:

            **iter**
                The iteration number.

            **rel red**
                The relative reduction of the solution from one iteration to the previous.
                Is only displayed from the second iteration onwards.

            **time**
                Seconds taken for the iteration.

            **resid**
                Residual of the last time step of the iteration.

            **err red**
                Reduction of the absolute error from the first iteration to the current.
                Is only displayed from the second iteration onwards and only if the given problem
                provides a function of the exact solution (see :py:meth:`.problem_has_exact_solution()`).

        The output for the time steps of an iteration explained (will only show up when running with
        logger level ``DEBUG``):

            **step**
                Number of the time step.

            **t_0**
                Start of the time step interval.

            **t_1**
                End of the time step interval.

            **resid**
                Residual of the time step.

            **err**
                Inifnity norm of error for the time step.
                Is only displayed if the given problem provides a function for the
                exact solution (see :py:meth:`.problem_has_exact_solution()`).

        Parameters
        ----------
        core : :py:class:`.SdcSolverCore`
            core solver stepping method
        dt : :py:class:`float`
            width of the interval to work on; this is devided into the number of given
            time steps this solver has been initialized with

        See Also
        --------
        :py:meth:`.IIterativeTimeSolver.run` : overridden method
        """
        super(ParallelSdc, self).run(core, **kwargs)

        assert_is_key(kwargs, 'dt', "Width of interval must be given", self)
        assert_is_instance(kwargs['dt'], float,
                           "Width of interval must be a float: NOT %s" % kwargs['dt'].__class__.__name__,
                           self)
        self._dt = kwargs['dt']

        self._print_header()

        # start iterations
        # TODO: exact solution storage handling
        self.__exact[0] = self.problem.initial_value

        _has_work = True
        _previous_flag = Message.SolverFlag.none
        _current_flag = Message.SolverFlag.none
        __work_loop_count = 1

        while _has_work:
            LOG.debug("Work Loop: %d" % __work_loop_count)
            _previous_flag = _current_flag
            _current_flag = Message.SolverFlag.none

            # receive dedicated message
            _msg = self._communicator.receive()

            if _msg.flag == Message.SolverFlag.failed:
                # previous solver failed
                # --> pass on the failure and abort
                _current_flag = Message.SolverFlag.failed
                _has_work = False
                LOG.debug("Previous Solver Failed")
            else:
                if _msg.flag == Message.SolverFlag.time_adjusted:
                    # the previous solver has adjusted its interval
                    # --> we need to recompute our interval
                    _current_flag = self._adjust_interval_width()
                    # we don't immediately start the computation of the newly computed interval
                    # but try to pass the new interval end to the next solver as soon as possible
                    # (this should avoid throwing away useless computation)
                    LOG.debug("Previous Solver Adjusted Time")
                else:
                    if _previous_flag in \
                            [Message.SolverFlag.none, Message.SolverFlag.converged, Message.SolverFlag.finished,
                             Message.SolverFlag.time_adjusted]:
                        # we just started or finished our previous interval
                        # --> start a new interval
                        _has_work = self._init_new_interval(_msg.time_point)

                        if _has_work:
                            # set initial values
                            self.state.initial.solution.value = _msg.value.copy()
                            self.state.initial.solution.time_point = _msg.time_point
                            self.state.initial.done()

                            LOG.debug("New Interval Initialized")

                            # start logging output
                            self._print_interval_header()

                            # start global timing (per interval)
                            self.timer.start()
                        else:
                            # pass
                            LOG.debug("No New Interval Available")
                    elif _previous_flag == Message.SolverFlag.iterating:
                        LOG.debug("Next Iteration")
                    else:
                        LOG.warn("WARNING!!! Something went wrong here")

                    if _has_work:
                        # we are still on the same interval or have just successfully initialized a new interval
                        # --> do the real computation
                        LOG.debug("Starting New Solver Main Loop")

                        # initialize a new iteration state
                        self.state.proceed()

                        if _msg.time_point == self.state.initial.time_point:
                            if _previous_flag == Message.SolverFlag.iterating:
                                LOG.debug("Updating initial value")
                                # if the previous solver has a new initial value for us, we use it
                                self.state.current_iteration.initial.solution.value = _msg.value.copy()

                        _current_flag = self._main_solver_loop()

                        if _current_flag in \
                                [Message.SolverFlag.converged, Message.SolverFlag.finished, Message.SolverFlag.failed]:
                            if self.state.last_iteration_index <= self.threshold.max_iterations:
                                LOG.info("> Converged after {:d} iteration(s).".format(self.state.last_iteration_index + 1))
                                LOG.info(">   {:s}".format(self.threshold.has_reached(human=True)))
                                LOG.info(">   Final Residual: {:.3e}"
                                         .format(supremum_norm(self.state.last_iteration.final_step.solution.residual)))
                                LOG.info(">   Solution Reduction: {:.3e}"
                                         .format(supremum_norm(self.state.solution.solution_reduction(self.state.last_iteration_index))))
                                if problem_has_exact_solution(self.problem, self):
                                    LOG.info(">   Error Reduction: {:.3e}"
                                             .format(supremum_norm(self.state.solution.error_reduction(self.state.last_iteration_index))))
                            else:
                                warnings.warn("{} SDC: Did not converged: {:s}".format(self._core.name, self.problem))
                                LOG.info("> FAILED: After maximum of {:d} iteration(s).".format(self.state.last_iteration_index + 1))
                                LOG.info(">   Final Residual: {:.3e}"
                                         .format(supremum_norm(self.state.last_iteration.final_step.solution.residual)))
                                LOG.info(">   Solution Reduction: {:.3e}"
                                         .format(supremum_norm(self.state.solution.solution_reduction(self.state.last_iteration_index))))
                                if problem_has_exact_solution(self.problem, self):
                                    LOG.info(">   Error Reduction: {:.3e}"
                                             .format(supremum_norm(self.state.solution.error_reduction(self.state.last_iteration_index))))
                                LOG.warn("{} SDC Failed: Maximum number iterations reached without convergence.".format(self._core.name))
                    elif _previous_flag in [Message.SolverFlag.converged, Message.SolverFlag.finished]:
                        LOG.debug("Solver Finished.")

                        self.timer.stop()

                        self._print_footer()
                    else:
                        # something went wrong
                        # --> we failed
                        LOG.warn("Solver failed.")
                        _current_flag = Message.SolverFlag.failed

            self._communicator.send(value=self.state.current_iteration.final_step.solution.value,
                                    time_point=self.state.current_iteration.final_step.time_point,
                                    flag=_current_flag)
            __work_loop_count += 1

        # end while:has_work is None
        LOG.debug("Solver Main Loop Done")

        return [_s.solution for _s in self._states]

    @property
    def state(self):
        """Read-only accessor for the sovler's state

        Returns
        -------
        state : :py:class:`.ISolverState`
        """
        return self._states[-1]

    @property
    def num_time_steps(self):
        """Accessor for the number of time steps within the interval.

        Returns
        -------
        number_time_steps : :py:class:`int`
            Number of time steps within the problem-given time interval.
        """
        return self._num_time_steps

    @property
    def num_nodes(self):
        """Accessor for the number of integration nodes per time step.

        Returns
        -------
        number_of_nodes : :py:class:`int`
            Number of integration nodes used within one time step.
        """
        return self.__num_nodes

    def _init_new_state(self):
        """Initialize a new state for a work task

        Usually, this starts a new work task.
        The previous state, if applicable, is stored in a stack.
        """
        if self.state:
            # finalize the current state
            self.state.finalize()

        # initialize solver state
        self._states.append(SdcSolverState(num_nodes=self.num_nodes - 1, num_time_steps=self.num_time_steps))

    def _init_new_interval(self, start):
        """Initializes a new work interval

        Parameters
        ----------
        start : :py:class:`float`
            start point of new interval

        Returns
        -------
        has_work : :py:class:`bool`

            :py:class:`True`
                if new interval have been initialized
            :py:class:`False`
                if no new interval have been initialized
                (i.e. new interval end would exceed end of time given by problem)
        """
        assert_is_instance(start, float,
                           "Time point must be a float: NOT %s" % start.__class__.__name__,
                           self)

        if start + self._dt > self.problem.time_end:
            return False

        if start == self.state.initial.time_point:
            return False

        self._init_new_state()

        # set width of current interval
        self.state.delta_interval = self._dt

        # compute time step and node distances
        self._deltas['t'] = self.state.delta_interval / self.num_time_steps  # width of a single time step (equidistant)

        # start time points of time steps
        self.__time_points['steps'] = np.linspace(start, start + self._dt, self.num_time_steps + 1)

        # initialize and transform integrator for time step width
        self._integrator.init(self.__nodes_type(), self.__num_nodes, self.__weights_type(),
                              interval=np.array([self.__time_points['steps'][0], self.__time_points['steps'][1]],
                                                dtype=np.float))

        self.__time_points['nodes'] = np.zeros((self.num_time_steps, self.num_nodes), dtype=np.float)
        _deltas_n = np.zeros(self.num_time_steps * (self.num_nodes - 1) + 1)

        # copy the node provider so we do not alter the integrator's one
        _nodes = deepcopy(self._integrator.nodes_type)
        for _t in range(0, self.num_time_steps):
            # transform Nodes (copy) onto new time step for retrieving actual integration nodes
            _nodes.interval = np.array([self.__time_points['steps'][_t], self.__time_points['steps'][_t + 1]])
            self.__time_points['nodes'][_t] = _nodes.nodes.copy()
            for _n in range(0, self.num_nodes - 1):
                _i = _t * (self.num_nodes - 1) + _n
                _deltas_n[_i + 1] = _nodes.nodes[_n + 1] - _nodes.nodes[_n]
        self._deltas['n'] = _deltas_n[1:].copy()

        return True

    def _adjust_interval_width(self):
        """Adjust width of time interval
        """
        raise NotImplementedError("Time Adaptivity not yet implemented.")
        # return Message.SolverFlag.time_adjusted

    def _main_solver_loop(self):
        # initialize iteration timer of same type as global timer
        _iter_timer = self.timer.__class__()

        self._print_iteration(self.state.current_iteration_index + 1)

        # iterate on time steps
        _iter_timer.start()
        for _current_time_step in self.state.current_iteration:
            # run this time step
            self._time_step()
            if self.state.current_time_step_index < len(self.state.current_iteration) - 1:
                self.state.current_iteration.proceed()
        _iter_timer.stop()

        # check termination criteria
        self.threshold.check(self.state)

        # log this iteration's summary
        if self.state.is_first_iteration:
            # on first iteration we do not have comparison values
            self._print_iteration_end(None, None, None, _iter_timer.past())
        else:
            if problem_has_exact_solution(self.problem, self) and not self.state.is_first_iteration:
                # we could compute the correct error of our current solution
                self._print_iteration_end(self.state.solution.solution_reduction(self.state.current_iteration_index),
                                          self.state.solution.error_reduction(self.state.current_iteration_index),
                                          self.state.current_step.solution.residual,
                                          _iter_timer.past())
            else:
                self._print_iteration_end(self.state.solution.solution_reduction(self.state.current_iteration_index),
                                          None,
                                          self.state.current_step.solution.residual,
                                          _iter_timer.past())

        # finalize this iteration (i.e. TrajectorySolutionData.finalize())
        self.state.current_iteration.finalize()

        _reason = self.threshold.has_reached()
        if _reason is None:
            # LOG.debug("solver main loop done: no reason")
            return Message.SolverFlag.iterating
        elif _reason == ['iterations']:
            # LOG.debug("solver main loop done: iterations")
            return Message.SolverFlag.finished
        else:
            # LOG.debug("solver main loop done: other")
            return Message.SolverFlag.converged

    def _time_step(self):
        self.state.current_time_step.delta_time_step = self._deltas['t']
        for _step in range(0, len(self.state.current_time_step)):
            _node_index = self.state.current_time_step_index * (self.num_nodes - 1) + _step
            self.state.current_time_step[_step].delta_tau = self._deltas['n'][_node_index]
            self.state.current_time_step[_step].solution.time_point = \
                self.__time_points['nodes'][self.state.current_time_step_index][_step + 1]

        self._print_time_step(self.state.current_time_step_index + 1,
                              self.state.current_time_step.initial.time_point,
                              self.state.current_time_step.last.time_point,
                              self.state.current_time_step.delta_time_step)

        for _current_step in self.state.current_time_step:
            self._sdc_step()
            if self.state.current_step_index < len(self.state.current_time_step) - 1:
                self.state.current_time_step.proceed()

        self._print_time_step_end()

        # finalizing the current time step (i.e. TrajectorySolutionData.finalize)
        self.state.current_time_step.finalize()

    def _sdc_step(self):
        # helper variables
        _current_time_step_index = self.state.current_time_step_index
        _current_step_index = self.state.current_step_index

        # copy solution of previous iteration to this one
        if self.state.is_first_iteration:
            self.state.current_step.solution.value = self.state.initial.solution.value.copy()
        else:
            self.state.current_step.solution.value = \
                self.state.previous_iteration[_current_time_step_index][_current_step_index].solution.value.copy()

        # gather values for integration and evaluate problem at given points
        #  initial value for this time step
        _integrate_values = \
            np.array(
                [self.problem.evaluate(self.state.current_time_step.initial.solution.time_point,
                                       self.state.current_time_step.initial.solution.value.copy())
                ], dtype=self.problem.numeric_type)

        if _current_step_index > 0:
            #  values from this iteration (already calculated)
            _from_current_iteration_range = range(0, _current_step_index)
            for _index in _from_current_iteration_range:
                _integrate_values = \
                    np.append(_integrate_values,
                              np.array(
                                  [self.problem.evaluate(self.state.current_time_step[_index].solution.time_point,
                                                         self.state.current_time_step[_index].solution.value.copy())
                                  ], dtype=self.problem.numeric_type
                              ), axis=0)

        #  values from previous iteration
        _from_previous_iteration_range = range(_current_step_index, self.num_nodes - 1)
        for _index in _from_previous_iteration_range:
            if self.state.is_first_iteration:
                _this_value = self.state.initial.solution.value
            else:
                _this_value = self.state.previous_iteration[_current_time_step_index][_index].solution.value.copy()
            _integrate_values = \
                np.append(_integrate_values,
                          np.array(
                              [self.problem.evaluate(self.state.current_time_step[_index].solution.time_point,
                                                     _this_value)
                              ], dtype=self.problem.numeric_type
                          ), axis=0)
        assert_condition(_integrate_values.size == self.num_nodes,
                         ValueError, "Number of integration values not correct: {:d} != {:d}"
                         .format(_integrate_values.size, self.num_nodes),
                         self)

        # integrate
        self.state.current_step.integral = self._integrator.evaluate(_integrate_values,
                                                                     last_node_index=_current_step_index + 1)

        # compute step
        self._core.run(self.state, problem=self.problem)

        # calculate residual
        _integrate_values[_current_step_index] = \
            np.array(
                [self.problem.evaluate(self.state.current_step.solution.time_point,
                                       self.state.current_step.solution.value.copy())
                ], dtype=self.problem.numeric_type)

        _residual_integral = np.zeros(self.problem.dim, dtype=self.problem.numeric_type)
        for i in range(0, _current_step_index + 1):
            _residual_integral += self._integrator.evaluate(_integrate_values, last_node_index=i + 1)
        del _integrate_values

        self._core.compute_residual(self.state, integral=_residual_integral)

        # calculate error
        self._core.compute_error(self.state, problem=self.problem)

        # log
        _previous_time = self.state.previous_step.time_point

        if problem_has_exact_solution(self.problem, self):
            self._print_step(self.state.current_step_index + 2,
                             _previous_time,
                             self.state.current_step.time_point,
                             supremum_norm(self.state.current_step.solution.value),
                             self.state.current_step.solution.residual,
                             self.state.current_step.solution.error)
        else:
            self._print_step(self.state.current_step_index + 2,
                             _previous_time,
                             self.state.current_step.time_point,
                             supremum_norm(self.state.current_step.solution.value),
                             self.state.current_step.solution.residual,
                             None)

        # finalize this current step (i.e. StepSolutionData.finalize())
        self.state.current_step.done()

    def _print_header(self):
        LOG.info("> " + '#' * 78)
        LOG.info("{:#<80}".format("> START: {:s} ".format(self._core.name)))
        LOG.info(">   Time Steps:             {:d}".format(self.num_time_steps))
        LOG.info(">   Integration Nodes:      {:d}".format(self.num_nodes))
        LOG.info(">   Termination Conditions: {:s}".format(self.threshold.print_conditions()))
        LOG.info(">   Problem: {:s}".format(self.problem))

    def _print_interval_header(self):
        LOG.info("> " + '-' * 78)
        LOG.info(">   Interval:               [{:.3f}, {:.3f}]".format(self.state.initial.time_point,
                                                                       self.state.initial.time_point + self._dt))
        self._print_output_tree_header()

    def _print_output_tree_header(self):
        LOG.info(">    iter")
        LOG.info(">         \\")
        LOG.info("!>          |- time    start     end        delta")
        LOG.info("!>          |     \\")
        LOG.info("!>          |      |- step    t_0      t_1       phi(t_1)   resid      err")
        LOG.info("!>          |      \\_")
        LOG.info(">          \\_   sol r.red    err r.red      resid       time")

    def _print_iteration(self, iter):
        _iter = self._output_format(iter, 'int', width=5)
        LOG.info(">    %s" % _iter)
        LOG.info(">         \\")

    def _print_iteration_end(self, solred, errred, resid, time):
        _solred = self._output_format(solred, 'exp')
        _errred = self._output_format(errred, 'exp')
        _resid = self._output_format(resid, 'exp')
        _time = self._output_format(time, 'float', width=6.3)
        LOG.info(">          \\_   %s    %s    %s    %s" % (_solred, _errred, _resid, _time))

    def _print_time_step(self, time_step, start, end, delta):
        _time_step = self._output_format(time_step, 'int', width=3)
        _start = self._output_format(start, 'float', width=6.3)
        _end = self._output_format(end, 'float', width=6.3)
        _delta = self._output_format(delta, 'float', width=6.3)
        LOG.info("!>          |- %s    %s    %s    %s" % (_time_step, _start, _end, _delta))
        LOG.info("!>          |     \\")
        self._print_step(1, None, self.state.current_time_step.initial.time_point,
                         supremum_norm(self.state.current_time_step.initial.solution.value),
                         None, None)

    def _print_time_step_end(self):
        LOG.info("!>          |      \\_")

    def _print_step(self, step, t0, t1, phi, resid, err):
        _step = self._output_format(step, 'int', width=2)
        _t0 = self._output_format(t0, 'float', width=6.3)
        _t1 = self._output_format(t1, 'float', width=6.3)
        _phi = self._output_format(phi, 'float', width=6.3)
        _resid = self._output_format(resid, 'exp')
        _err = self._output_format(err, 'exp')
        LOG.info("!>          |      |- %s    %s    %s    %s    %s    %s" % (_step, _t0, _t1, _phi, _resid, _err))

    def _print_footer(self):
        LOG.info("{:#<80}".format("> FINISHED: {:s} SDC ({:.3f} sec) ".format(self._core.name, self.timer.past())))
        LOG.info("> " + '#' * 78)

    def _output_format(self, value, type, width=None):
        def _value_to_numeric(value):
            if isinstance(value, (np.ndarray, IDiagnosisValue)):
                return supremum_norm(value)
            else:
                return value

        if type and width is None:
            if type == 'float':
                width = 10.3
            elif type == 'int':
                width = 10
            elif type == 'exp':
                width = 9.2
            else:
                width = 10

        if value is None:
            _outstr = "{: ^{width}s}".format('na', width=int(width))
        else:
            if type == 'float':
                _outstr = "{: {width}f}".format(_value_to_numeric(value), width=width)
            elif type == 'int':
                _outstr = "{: {width}d}".format(_value_to_numeric(value), width=width)
            elif type == 'exp':
                _outstr = "{: {width}e}".format(_value_to_numeric(value), width=width)
            else:
                _outstr = "{: >{width}s}".format(value, width=width)

        return _outstr


__all__ = ['Sdc']
