# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from copy import deepcopy
import warnings as warnings
from collections import OrderedDict

import numpy as np

from pypint.solvers.i_iterative_time_solver import IIterativeTimeSolver
from pypint.solvers.i_parallel_solver import IParallelSolver
from pypint.communicators.message import Message
from pypint.multi_level_providers.multi_time_level_provider import MultiTimeLevelProvider
from pypint.integrators.integrator_base import IntegratorBase
from pypint.integrators.node_providers.gauss_lobatto_nodes import GaussLobattoNodes
from pypint.integrators.weight_function_providers.polynomial_weight_function import PolynomialWeightFunction
from pypint.problems import IInitialValueProblem, problem_has_exact_solution
from pypint.solvers.states.mlsdc_solver_state import MlSdcSolverState
from pypint.solvers.diagnosis import IDiagnosisValue
from pypint.solvers.diagnosis.norms import supremum_norm
from pypint.plugins.timers.timer_base import TimerBase
from pypint.utilities.threshold_check import ThresholdCheck
from pypint.utilities import assert_is_instance, assert_condition, func_name, assert_named_argument
from pypint.utilities.logging import *


class MlSdc(IIterativeTimeSolver, IParallelSolver):
    """
    See Also
    --------
    :py:class:`.IIterativeTimeSolver` :
        implemented interface
    :py:class:`.IParallelSolver` :
        mixed-in interface
    """
    def __init__(self, **kwargs):
        super(MlSdc, self).__init__(**kwargs)
        IParallelSolver.__init__(self, **kwargs)
        del self._state
        del self._integrator

        self.threshold = ThresholdCheck(min_threshold=1e-7, max_threshold=10, conditions=("residual", "iterations"))
        self.timer = TimerBase()

        self._dt = 0.0
        self._ml_provider = None

        self.__nodes_type = GaussLobattoNodes
        self.__weights_type = PolynomialWeightFunction
        self.__exact = np.zeros(0)
        self.__deltas = None  # deltas between nodes as array; for each level (0: coarsest)
        self.__time_points = None  # time points of nodes as array; for each level

    def init(self, problem, **kwargs):
        """Initializes MLSDC solver with given problem, integrator and multi-level provider.

        Parameters
        ----------
        ml_provider : :py:class:`.MultiLevelProvider`
            *(required)*
            handler for the different levels to use
        num_nodes : :py:class:`int`
            *(otional)*
            number of nodes per time step
        nodes_type : :py:class:`.INodes`
            *(optional)*
            Type of integration nodes to be used (class name, **NOT instance**).
        weights_type : :py:class:`.IWeightFunction`
            *(optional)*
            Integration weights function to be used (class name, **NOT instance**).

        Raises
        ------
        ValueError :

            * if given problem is not an :py:class:`.IInitialValueProblem`
            * if number of nodes per time step is not given; neither through ``num_nodes``, ``nodes_type`` nor
              ``integrator``
            * if no :py:class:`.MultiLevelProvider` is given

        See Also
        --------
        :py:meth:`.IIterativeTimeSolver.init`
            overridden method (with further parameters)
        :py:meth:`.IParallelSolver.init`
            mixed in overridden method (with further parameters)
        """
        assert_is_instance(problem, IInitialValueProblem, descriptor="Initial Value Problem", checking_obj=self)

        assert_named_argument('ml_provider', kwargs, types=MultiTimeLevelProvider,
                              descriptor='Multi Time Level Provider', checking_obj=self)
        self._ml_provider = kwargs['ml_provider']

        super(MlSdc, self).init(problem, **kwargs)

        # TODO: need to store the exact solution somewhere else
        self.__exact = np.zeros(self.ml_provider.integrator(-1).num_nodes, dtype=np.object)

    def run(self, core, **kwargs):
        """Applies SDC solver to the initialized problem setup.

        Solves the given problem with the explicit SDC algorithm.

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
        super(MlSdc, self).run(core, **kwargs)

        assert_named_argument('dt', kwargs, types=float, descriptor="Width of Interval", checking_obj=self)
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
            # LOG.debug("Work Loop: %d" % __work_loop_count)
            _previous_flag = _current_flag
            _current_flag = Message.SolverFlag.none

            # receive dedicated message
            _msg = self._communicator.receive(tag=(self.ml_provider.num_levels - 1))

            if _msg.flag == Message.SolverFlag.failed:
                # previous solver failed
                # --> pass on the failure and abort
                _current_flag = Message.SolverFlag.failed
                _has_work = False
                # LOG.debug("Previous Solver Failed")
            else:
                if _msg.flag == Message.SolverFlag.time_adjusted:
                    # the previous solver has adjusted its interval
                    # --> we need to recompute our interval
                    _current_flag = self._adjust_interval_width()
                    # we don't immediately start the computation of the newly computed interval
                    # but try to pass the new interval end to the next solver as soon as possible
                    # (this should avoid throwing away useless computation)
                    # LOG.debug("Previous Solver Adjusted Time")
                else:
                    if _previous_flag in \
                            [Message.SolverFlag.none, Message.SolverFlag.converged, Message.SolverFlag.finished,
                             Message.SolverFlag.time_adjusted]:
                        # we just started or finished our previous interval
                        # --> start a new interval
                        _has_work = self._init_new_interval(_msg.time_point)

                        if _has_work:
                            # set initial values
                            self.state.initial.value = _msg.value.copy()
                            self.state.initial.solution.time_point = _msg.time_point
                            self.state.initial.done()

                            # LOG.debug("New Interval Initialized")

                            # start logging output
                            self._print_interval_header()

                            # start global timing (per interval)
                            self.timer.start()
                        else:
                            # LOG.debug("No New Interval Available")
                            pass
                    elif _previous_flag == Message.SolverFlag.iterating:
                        # LOG.debug("Next Iteration")
                        pass
                    else:
                        # LOG.warn("WARNING!!! Something went wrong here")
                        pass

                    if _has_work:
                        # we are still on the same interval or have just successfully initialized a new interval
                        # --> do the real computation
                        # LOG.debug("Starting New Solver Main Loop")

                        # initialize a new iteration state
                        self.state.proceed()

                        if _msg.time_point == self.state.initial.time_point:
                            if _previous_flag == Message.SolverFlag.iterating:
                                # LOG.debug("Updating initial value")
                                # if the previous solver has a new initial value for us, we use it
                                self.state.current_iteration.initial.value = _msg.value.copy()

                        _current_flag = self._main_solver_loop()

                        if _current_flag in \
                                [Message.SolverFlag.converged, Message.SolverFlag.finished, Message.SolverFlag.failed]:
                            _log_msgs = {'': OrderedDict()}
                            if self.state.last_iteration_index <= self.threshold.max_iterations:
                                _group = 'Converged after %d iteration(s)' % (self.state.last_iteration_index + 1)
                                _log_msgs[''][_group] = OrderedDict()
                                _log_msgs[''][_group] = self.threshold.has_reached(log=True)
                                _log_msgs[''][_group]['Final Residual'] = "{:.3e}"\
                                    .format(supremum_norm(self.state.last_iteration.final_step.solution.residual))
                                _log_msgs[''][_group]['Solution Reduction'] = "{:.3e}"\
                                    .format(supremum_norm(self.state.solution
                                                          .solution_reduction(self.state.last_iteration_index)))
                                if problem_has_exact_solution(self.problem, self):
                                    _log_msgs[''][_group]['Error Reduction'] = "{:.3e}"\
                                        .format(supremum_norm(self.state.solution
                                                              .error_reduction(self.state.last_iteration_index)))
                            else:
                                warnings.warn("{}: Did not converged: {:s}".format(self._core.name, self.problem))
                                _group = "FAILED: After maximum of {:d} iteration(s)"\
                                         .format(self.state.last_iteration_index + 1)
                                _log_msgs[''][_group] = OrderedDict()
                                _log_msgs[''][_group]['Final Residual'] = "{:.3e}"\
                                    .format(supremum_norm(self.state.last_iteration.final_step.solution.residual))
                                _log_msgs[''][_group]['Solution Reduction'] = "{:.3e}"\
                                    .format(supremum_norm(self.state.solution
                                                          .solution_reduction(self.state.last_iteration_index)))
                                if problem_has_exact_solution(self.problem, self):
                                    _log_msgs[''][_group]['Error Reduction'] = "{:.3e}"\
                                        .format(supremum_norm(self.state.solution
                                                              .error_reduction(self.state.last_iteration_index)))
                                LOG.warn("  {} Failed: Maximum number iterations reached without convergence."
                                         .format(self._core.name))
                            print_logging_message_tree(_log_msgs)
                    elif _previous_flag in [Message.SolverFlag.converged, Message.SolverFlag.finished]:
                        # LOG.debug("Solver Finished.")

                        self.timer.stop()

                        self._print_footer()
                    else:
                        # something went wrong
                        # --> we failed
                        # LOG.warn("Solver failed.")
                        _current_flag = Message.SolverFlag.failed

            self._communicator.send(value=self.state.current_iteration.finest_level.final_step.value,
                                    time_point=self.state.current_iteration.finest_level.final_step.time_point,
                                    flag=_current_flag)
            __work_loop_count += 1

        # end while:has_work is None
        # LOG.debug("Solver Main Loop Done")

        return [_s.solution for _s in self._states]

    @property
    def state(self):
        """Read-only accessor for the sovler's state

        Returns
        -------
        state : :py:class:`.ISolverState`
        """
        if len(self._states) > 0:
            return self._states[-1]
        else:
            return None

    @property
    def ml_provider(self):
        """Read-only accessor for the multi level provider

        Returns
        -------
        multi_level_provider : :py:class:`.MultiLevelProvider`
        """
        return self._ml_provider

    def _main_solver_loop(self):
        # initialize iteration timer of same type as global timer
        _iter_timer = self.timer.__class__()

        # initialize solver states for this iteration
        self._init_new_iteration()

        self._print_iteration(self.state.current_iteration_index + 1)

        # iterate on time steps
        _iter_timer.start()
        self._level()
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

    def _init_new_state(self):
        """Initialize a new state for a work task

        Usually, this starts a new work task.
        The previous state, if applicable, is stored in a stack.
        """
        if self.state:
            # print("Finished a State")
            # finalize the current state
            self.state.finalize()

        # print("Stating a new state")
        # initialize solver state
        self._states.append(MlSdcSolverState(num_level=self.ml_provider.num_levels))

    def _init_new_interval(self, start):
        """Initializes a new work interval

        Parameters
        ----------
        start : :py:class:`float`
            start point of new interval

        Returns
        -------
        has_work : :py:class:`bool`
            :py:class:`True` if new interval have been initialized;
            :py:class:`False` if no new interval have been initialized (i.e. new interval end would exceed end of time
            given by problem)
        """
        assert_is_instance(start, float, descriptor="Time Point", checking_obj=self)

        if start + self._dt > self.problem.time_end:
            return False

        if self.state and start == self.state.initial.time_point:
            return False

        self._init_new_state()

        # set width of current interval
        self.state.initial.solution.time_point = start
        self.state.initial.value = self.problem.initial_value.copy()
        self.state.delta_interval = self._dt

        self.__time_points = np.zeros(self.ml_provider.num_levels, dtype=np.object)
        self.__deltas = np.zeros(self.ml_provider.num_levels, dtype=np.object)

        # make sure the integrators are all set up correctly for the different levels
        for _level in range(0, self.ml_provider.num_levels):
            _integrator = self.ml_provider.integrator(_level)

            _integrator.transform_interval(self.state.interval)

            # print("nodes: %s" % _integrator.nodes)

            self.__time_points[_level] = np.zeros(_integrator.num_nodes, dtype=np.float)
            self.__deltas[_level] = np.zeros(_integrator.num_nodes, dtype=np.float)

            for _node in range(0, _integrator.num_nodes - 1):
                self.__time_points[_level] = deepcopy(_integrator.nodes)
                self.__deltas[_level][_node + 1] = _integrator.nodes[_node + 1] - _integrator.nodes[_node]

        # print("Time Points: %s" % self.__time_points)

        return True

    def _init_new_iteration(self):
        _current_state = self.state.current_iteration
        _previous_iteration = self.state.previous_iteration

        # set initial values
        for _level_index in range(0, self.ml_provider.num_levels):
            _current_state.add_finer_level(self.ml_provider.integrator(_level_index).num_nodes - 1)
            _level = _current_state.finest_level
            assert_condition(len(_level) == self.ml_provider.integrator(_level_index).num_nodes - 1,
                             RuntimeError, "Number of Steps on Level %d not correct (%d)"
                             % (len(_level), self.ml_provider.integrator(_level_index).num_nodes - 1),
                             checking_obj=self)

            _level.initial = deepcopy(self.state.initial)
            if _previous_iteration is None:
                _level.broadcast(_level.initial.value)

            for _step_index in range(0, len(_level)):
                _level[_step_index].delta_tau = self.__deltas[_level_index][_step_index + 1]
                _level[_step_index].solution.time_point = self.__time_points[_level_index][_step_index + 1]
                if _previous_iteration is not None:
                    _level[_step_index].value = _previous_iteration[_level_index][_step_index].value.copy()

        assert_condition(len(self.state.current_iteration) == self.ml_provider.num_levels,
                         RuntimeError, "Number of levels in current state not correct."
                                       " (this shouldn't have happend)",
                         checking_obj=self)

    def _adjust_interval_width(self):
        """Adjust width of time interval
        """
        raise NotImplementedError("Time Adaptivity not yet implemented.")
        # return Message.SolverFlag.time_adjusted

    def _compute_fas_correction(self, q_rhs_fine, fas_fine, q_rhs_coarse, fine_lvl):
        # add fas correction of finer level if available
        _fine_data = q_rhs_fine + fas_fine if fas_fine is not None else q_rhs_fine

        # 1. restringate fine data
        _restringated_fine = self.ml_provider.restringate(_fine_data, fine_lvl)
        LOG.debug("R x (Q_fine x F_fine + FAS_fine):\n    %s\n  = R(%s)\n  = R(%s + %s)"
                  % (_restringated_fine, _fine_data, q_rhs_fine, fas_fine))

        assert_condition(q_rhs_coarse.shape == _restringated_fine.shape,
                         ValueError,
                         message='Dimensions of coarse data and restringated fine data do not match: %s != %s'
                                 % (q_rhs_coarse.shape, _restringated_fine.shape),
                         checking_obj=self)

        # 2. compute difference (todo: with respect to interval length !?)
        self.state.current_iteration.current_level.fas_correction = _restringated_fine - q_rhs_coarse
        LOG.debug("FAS Correction:\n    %s\n  = %s - %s"
                  % (self.state.current_iteration.current_level.fas_correction, _restringated_fine, q_rhs_coarse))

    def _recompute_rhs_for_level(self, level):
        if level.rhs is None:
            if not level.initial.rhs_evaluated:
                level.initial.rhs = self.problem.evaluate(level.initial.time_point, level.initial.value)
            for step in level:
                if not step.rhs_evaluated:
                    step.rhs = self.problem.evaluate(step.time_point, step.value)

    def _compute_residual(self, finalize=False):
        LOG.debug("Computing Residual")
        self._print_step(1, None, self.state.current_level.initial.time_point,
                         supremum_norm(self.state.current_level.initial.value),
                         None, None)

        _full_integral = 0.0

        self._recompute_rhs_for_level(self.state.current_level)

        for _step_index in range(0, len(self.state.current_level)):
            _step = self.state.current_level[_step_index]

            if not _step.integral_available:
                _step.integral = \
                    self.ml_provider\
                        .integrator(self.state.current_level_index)\
                        .evaluate(self.state.current_level.rhs, from_node=_step_index, target_node=_step_index + 1)
            _full_integral += _step.integral

            self._core.compute_residual(self.state, step=_step, integral=_full_integral)

            if finalize:
                # finalize this step (i.e. StepSolutionData.finalize())
                _step.done()

        for _step_index in range(0, len(self.state.current_level)):
            _step = self.state.current_level[_step_index]
            if _step_index > 0:
                _previous_time = self.state.current_level[_step_index - 1].time_point
            else:
                _previous_time = self.state.current_level.initial.time_point

            if problem_has_exact_solution(self.problem, self):
                self._print_step(_step_index + 2,
                                 _previous_time,
                                 _step.time_point,
                                 supremum_norm(_step.value),
                                 _step.solution.residual,
                                 _step.solution.error)
            else:
                self._print_step(_step_index + 2,
                                 _previous_time,
                                 _step.time_point,
                                 supremum_norm(_step.value),
                                 _step.solution.residual,
                                 None)

        self._print_sweep_end()

    def _level(self):
        _current_level = self.state.current_iteration.current_level
        _finer_level = self.state.current_iteration.finer_level
        _coarser_level = self.state.current_iteration.coarser_level

        _msg = self.comm.receive(tag=self.state.current_level_index)
        if _msg and _msg.time_point == self.state.initial.time_point:
            _current_level.initial.definalize()
            _current_level.initial.value = _msg.value
            _current_level.initial.done()

        LOG.debug("Level %d initial values: %s"
                  % (self.state.current_iteration.current_level_index, _current_level.values.flatten()))

        self._print_level_header()

        self._recompute_rhs_for_level(_current_level)

        LOG.debug("Level %d RHS values: %s"
                  % (self.state.current_iteration.current_level_index, _current_level.rhs.flatten()))

        if not self.state.current_iteration.on_finest_level:
            # compute FAS Correction
            LOG.debug("Computing FAS Correction")
            _q_rhs_coarse = np.append(np.array([0.0], dtype=self.problem.numeric_type),
                                      np.array(
                                          [
                                              self.ml_provider
                                                  .integrator(self.state.current_iteration.current_level_index)
                                                  .evaluate(_current_level.rhs, target_node=_step_i+1)
                                              for _step_i in range(0, len(_current_level))
                                          ], dtype=self.problem.numeric_type))
            self._recompute_rhs_for_level(_finer_level)
            _q_rhs_fine = np.append(np.array([0.0], dtype=self.problem.numeric_type),
                                    np.array(
                                        [
                                            self.ml_provider
                                                .integrator(self.state.current_iteration.finer_level_index)
                                                .evaluate(_finer_level.rhs, target_node=_step_i+1)
                                            for _step_i in range(0, len(_finer_level))
                                        ], dtype=self.problem.numeric_type))

            self._compute_fas_correction(_q_rhs_fine, _finer_level.fas_correction, _q_rhs_coarse,
                                         fine_lvl=self.state.current_iteration.finer_level_index)

        if (not self.state.current_iteration.on_base_level and not self.state.current_iteration.on_finest_level) or \
                (self.state.current_iteration.on_finest_level and self.state.is_first_iteration):
            # pre-sweep
            LOG.debug("pre-sweep")
            self._sdc_sweep(copy=self.state.current_iteration.on_finest_level, with_residual=True)

        if not self.state.current_iteration.on_base_level:
            # restrict
            _coarser_level.values = \
                self.ml_provider.restringate(_current_level.values,
                                             fine_level=self.state.current_iteration.current_level_index,
                                             coarse_level=self.state.current_iteration.coarser_level_index)
            # call next coarser level
            self.state.current_iteration.step_down()
            #  RECURSION HERE!
            self._level()
            # -> coarser level is done; coming up again

            # coarse correction
            # TODO: correct RHS evaluations; not values
            _prolongated_coarse_correction = \
                self.ml_provider.prolongate(_coarser_level.coarse_corrections,
                                            fine_level=self.state.current_iteration.current_level_index,
                                            coarse_level=self.state.current_iteration.coarser_level_index)
            LOG.debug("Apply Coarse Correction\n  ==> %s\n    = %s + %s"
                      % ((_current_level.values + _prolongated_coarse_correction).flatten(),
                         _current_level.values.flatten(), _prolongated_coarse_correction.flatten()))
            _corrected_values = _current_level.values + _prolongated_coarse_correction
            for _step_index in range(0, len(_current_level)):
                _current_level[_step_index].intermediate.value = _corrected_values[_step_index + 1]

            LOG.debug("Recompute Errors")
            for _step_index in range(0, len(_current_level)):
                self._core.compute_error(self.state, step_index=_step_index, problem=self.problem)

            # post-sweep
            LOG.debug("post-sweep with coarse-corrected intermediate")
            self._sdc_sweep(use_intermediate=True, copy=False, with_residual=False)
        else:
            # post-sweep
            LOG.debug("post-sweep")
            self._sdc_sweep(use_intermediate=False, copy=False, with_residual=False)

        if not self.state.current_iteration.on_finest_level:
            # compute coarse correction
            LOG.debug("Computing Coarse Correction")
            _restringated_values = \
                self.ml_provider\
                    .restringate(_finer_level.values,
                                 self.state.current_iteration.finer_level_index,
                                 self.state.current_iteration.current_level_index)
            for _step_index in range(0, len(_current_level)):
                _step = _current_level[_step_index]
                _step.coarse_correction = _step.value - _restringated_values[_step_index + 1]
                LOG.debug("    %d: %s = %s - %s"
                          % (_step_index, _step.coarse_correction, _step.value, _restringated_values[_step_index + 1]))

        self._compute_residual(finalize=True)

        self.comm.send(tag=self.state.current_level_index,
                       value=_current_level.final_step.value,
                       time_point=_current_level.final_step.time_point)

        self._print_level_end()

        LOG.debug("Level %d final values: %s"
                  % (self.state.current_iteration.current_level_index, _current_level.values.flatten()))

        if not self.state.current_iteration.on_finest_level:
            # pass on to next finer level
            self.state.current_iteration.step_up()

    def _sdc_sweep(self, use_intermediate=False, copy=True, with_residual=False):
        """
        Parameters
        ----------
        copy : :py:class:`bool`
            passed on to :py:meth:`._sdc_step`
        with_residual : :py:class:`bool`
            whether to compute the residual at the end or not; *note: when the residual is computed, the sweep will be
            logged*
        """
        LOG.debug("Sweeping on level %d ..." % self.state.current_level_index)
        self.state.current_iteration.current_level.reset_to_start()
        _integrator = self.ml_provider.integrator(self.state.current_iteration.current_level_index)
        _num_nodes = _integrator.num_nodes

        # compute integral
        self.state.current_iteration.current_level.integral = 0.0

        if not self.state.current_iteration.current_level.initial.rhs_evaluated:
            self.state.current_iteration.current_level.initial.rhs = \
                self.problem.evaluate(self.state.current_iteration.current_level.initial.time_point,
                                      self.state.current_iteration.current_level.initial.value)

        _integrate_values = np.array([self.state.current_iteration.current_level.initial.rhs],
                                     dtype=self.problem.numeric_type)

        for _step_index in range(0, len(self.state.current_iteration.current_level)):
            # TODO: clean up this conditional
            if self.state.current_iteration.on_finest_level and self.state.is_first_iteration and not use_intermediate:
                LOG.debug("On First Iteration on Finest Level. Taking breadcasted initial value.")
                _integrate_values = \
                    np.append(_integrate_values,
                              np.array([self.state.current_iteration.current_level.initial.rhs],
                                       dtype=self.problem.numeric_type),
                              axis=0)

            elif not self.state.current_iteration.on_finest_level:
                LOG.debug("Not on Finest Level. Taking current intermediate value.")
                _step = self.state.current_iteration.current_level[_step_index]
                if use_intermediate:
                    if not _step.intermediate.rhs_evaluated:
                        _step.intermediate.rhs = self.problem.evaluate(_step.time_point, _step.intermediate.value)
                    _integrate_values = \
                        np.append(_integrate_values,
                                  np.array([_step.intermediate.rhs], dtype=self.problem.numeric_type),
                                  axis=0)
                else:
                    if not _step.rhs_evaluated:
                        _step.rhs = self.problem.evaluate(_step.time_point, _step.value)
                    _integrate_values = \
                        np.append(_integrate_values,
                                  np.array([_step.rhs], dtype=self.problem.numeric_type),
                                  axis=0)

            elif use_intermediate:
                LOG.debug("On Finest Level. Using intermediate value.")
                _step = self.state.current_iteration.current_level[_step_index]
                if not _step.intermediate.rhs_evaluated:
                    _step.intermediate.rhs = self.problem.evaluate(_step.time_point, _step.intermediate.value)
                _integrate_values = \
                    np.append(_integrate_values,
                              np.array([_step.intermediate.rhs], dtype=self.problem.numeric_type),
                              axis=0)

            else:
                LOG.debug("On Finest Level. Taking previous iteration's value.")
                _step = self.state.previous_iteration[self.state.current_iteration.current_level_index][_step_index]
                if use_intermediate:
                    if not _step.intermediate.rhs_evaluated:
                        _step.intermediate.rhs = self.problem.evaluate(_step.time_point, _step.intermediate.value)
                    _integrate_values = \
                        np.append(_integrate_values,
                                  np.array([_step.intermediate.rhs], dtype=self.problem.numeric_type),
                                  axis=0)
                else:
                    if not _step.rhs_evaluated:
                        _step.rhs = self.problem.evaluate(_step.time_point, _step.value)
                    _integrate_values = \
                        np.append(_integrate_values,
                                  np.array([_step.rhs], dtype=self.problem.numeric_type),
                                  axis=0)

        assert_condition(_integrate_values.size == _num_nodes,
                         ValueError, message="Number of integration values not correct: %d != %d"
                                             % (_integrate_values.size, _num_nodes),
                         checking_obj=self)

        LOG.debug("Integration Values: %s" % _integrate_values.flatten())
        if use_intermediate:
            LOG.debug("Values Before (intermediate): %s"
                      % ([_step.intermediate.value for _step in self.state.current_level]))
        else:
            LOG.debug("Values Before: %s" % self.state.current_iteration.current_level.values.flatten())

        # do the actual SDC steps of this SDC sweep
        for _step_index in range(0, len(self.state.current_iteration.current_level)):
            LOG.debug("Step %d:" % _step_index)
            _current_step = self.state.current_iteration.current_level[_step_index]
            # if not _current_step.integral_available:
            # TODO: fix unneccessary recomputation of integrals
            _current_step.integral = _integrator.evaluate(_integrate_values,
                                                          from_node=_step_index, target_node=_step_index + 1)

            # we successively compute the full integral
            LOG.debug("  Full Integral up to %d: %s = %s + %s"
                      % (_step_index + 1, self.state.current_iteration.current_level.integral + _current_step.integral,
                         self.state.current_iteration.current_level.integral, _current_step.integral))
            self.state.current_iteration.current_level.integral += _current_step.integral

            # do the SDC step of this sweep
            self._sdc_step(use_intermediate=use_intermediate, copy=copy)

            if self.state.current_level.current_step != self.state.current_level.final_step:
                self.state.current_level.proceed()

        del _integrate_values

        LOG.debug("Values After: %s" % self.state.current_iteration.current_level.values.flatten())

        if with_residual:
            self._compute_residual()

    def _sdc_step(self, use_intermediate=False, copy=True):
        """
        Parameters
        ----------
        copy : :py:class:`bool`
            whether to copy value from previous iteration to this one or just assume the value of the current iteration
            have been set before
        """
        if copy:
            # copy solution of previous iteration to this one
            if self.state.is_first_iteration:
                if use_intermediate:
                    LOG.debug("Coppying Initial Value to step %d intermediate" % self.state.current_step_index)
                    self.state.current_step.intermediate.value = self.state.current_level.initial.value.copy()
                else:
                    LOG.debug("Coppying Initial Value to step %d" % self.state.current_step_index)
                    self.state.current_step.value = self.state.current_level.initial.value.copy()
            else:
                if use_intermediate:
                    LOG.debug("Coppying Previous Iteration's Value to step %d intermediate"
                              % self.state.current_step_index)
                    self.state.current_step.intermediate.value = \
                        self.state.previous_iteration[self.state.current_level_index][self.state.current_step_index].value.copy()
                else:
                    LOG.debug("Coppying Previous Iteration's Value to step %d" % self.state.current_step_index)
                    self.state.current_step.value = \
                        self.state.previous_iteration[self.state.current_level_index][self.state.current_step_index].value.copy()

        # compute step
        self._core.run(self.state, problem=self.problem, use_intermediate=use_intermediate)

        # calculate error
        self._core.compute_error(self.state, problem=self.problem)

        # step gets finalized after computation of residual

    def print_lines_for_log(self):
        _lines = super(MlSdc, self).print_lines_for_log()
        return _lines

    def _print_interval_header(self):
        LOG.info("%s%s" % (VERBOSITY_LVL1, SEPARATOR_LVL3))
        LOG.info("{}  Interval: [{:.3f}, {:.3f}]"
                 .format(VERBOSITY_LVL1, self.state.initial.time_point, self.state.initial.time_point + self._dt))
        self._print_output_tree_header()

    def _print_output_tree_header(self):
        LOG.info("%s   iter" % VERBOSITY_LVL1)
        LOG.info("%s        \\" % VERBOSITY_LVL2)
        LOG.info("%s         |- level    nodes" % VERBOSITY_LVL2)
        LOG.info("%s         |     \\" % VERBOSITY_LVL3)
        LOG.info("%s         |      |- step    t_0      t_1       phi(t_1)    resid       err" % VERBOSITY_LVL3)
        LOG.info("%s         |      \\_" % VERBOSITY_LVL2)
        LOG.info("%s         \\_   sol r.red    err r.red      resid       time" % VERBOSITY_LVL1)

    def _print_iteration(self, _iter):
        _iter = self._output_format(_iter, 'int', width=4)
        LOG.info("%s   %s" % (VERBOSITY_LVL1, _iter))
        LOG.info("%s       \\" % VERBOSITY_LVL2)

    def _print_level_header(self):
        _lvl = self._output_format(self.state.current_level_index, 'int', width=2)
        _nodes = self._output_format(self.ml_provider.integrator(self.state.current_level_index).num_nodes,
                                     'int', width=2)
        LOG.info("%s        %s|- %s    %s" % (VERBOSITY_LVL2, ('|      ' * (self.ml_provider.num_levels - self.state.current_level_index - 1)),
                                               _lvl, _nodes))
        LOG.info("%s        %s|     \\" % (VERBOSITY_LVL3, ('|      ' * (self.ml_provider.num_levels - self.state.current_level_index - 1))))

    def _print_level_end(self):
        LOG.info("%s        %s|      \\_" % (VERBOSITY_LVL2, ('|      ' * (self.ml_provider.num_levels - self.state.current_level_index - 1))))

    def _print_iteration_end(self, solred, errred, resid, time):
        _solred = self._output_format(solred, 'exp')
        _errred = self._output_format(errred, 'exp')
        _resid = self._output_format(resid, 'exp')
        _time = self._output_format(time, 'float', width=6.3)
        LOG.info("%s        \\_   %s    %s    %s    %s" % (VERBOSITY_LVL1, _solred, _errred, _resid, _time))

    def _print_step(self, step, t0, t1, phi, resid, err):
        _step = self._output_format(step, 'int', width=2)
        _t0 = self._output_format(t0, 'float', width=6.3)
        _t1 = self._output_format(t1, 'float', width=6.3)
        _phi = self._output_format(phi, 'float', width=6.3)
        _resid = self._output_format(resid, 'exp')
        _err = self._output_format(err, 'exp')
        LOG.info("%s        %s|- %s    %s    %s    %s    %s    %s"
                 % (VERBOSITY_LVL3, ('|      ' * (self.ml_provider.num_levels - self.state.current_level_index)),
                    _step, _t0, _t1, _phi, _resid, _err))

    def _print_sweep_end(self):
        LOG.info("%s        %s|    \\_"
                 % (VERBOSITY_LVL3, ('|      ' * (self.ml_provider.num_levels - self.state.current_level_index))))

    def _output_format(self, value, _type, width=None):
        def _value_to_numeric(val):
            if isinstance(val, (np.ndarray, IDiagnosisValue)):
                return supremum_norm(val)
            else:
                return val

        if _type and width is None:
            if _type == 'float':
                width = 10.3
            elif _type == 'int':
                width = 10
            elif _type == 'exp':
                width = 9.2
            else:
                width = 10

        if value is None:
            _outstr = "{: ^{width}s}".format('na', width=int(width))
        else:
            if _type == 'float':
                _outstr = "{: {width}f}".format(_value_to_numeric(value), width=width)
            elif _type == 'int':
                _outstr = "{: {width}d}".format(_value_to_numeric(value), width=width)
            elif _type == 'exp':
                _outstr = "{: {width}e}".format(_value_to_numeric(value), width=width)
            else:
                _outstr = "{: >{width}s}".format(value, width=width)

        return _outstr


__all__ = ['MlSdc']
