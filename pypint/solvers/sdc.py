# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from copy import deepcopy
import warnings as warnings

import numpy as np

from pypint.solvers.i_iterative_time_solver import IIterativeTimeSolver
from pypint.integrators.sdc_integrator import SdcIntegrator
from pypint.integrators.node_providers.gauss_lobatto_nodes import GaussLobattoNodes
from pypint.integrators.weight_function_providers.polynomial_weight_function import PolynomialWeightFunction
from pypint.problems import IInitialValueProblem, problem_has_exact_solution
from pypint.solvers.states.sdc_solver_state import SdcSolverState
from pypint.solvers.diagnosis import IDiagnosisValue
from pypint.solvers.diagnosis.norms import supremum_norm
from pypint.plugins.timers.timer_base import TimerBase
from pypint.utilities.threshold_check import ThresholdCheck
from pypint.utilities import assert_is_instance, assert_condition, func_name
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


class Sdc(IIterativeTimeSolver):
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

    Examples
    --------
    >>> from pypint.solvers.sdc import Sdc
    >>> from pypint.solvers.cores import ExplicitSdcCore
    >>> from examples.problems.constant import Constant
    >>> # setup the problem
    >>> my_problem = Constant(constant=-1.0)
    >>> # create the solver
    >>> my_solver = Sdc()
    >>> # initialize the solver with the problem
    >>> my_solver.init(problem=my_problem, num_time_steps=1, num_nodes=3)
    >>> # run the solver and get the solution
    >>> my_solution = my_solver.run(ExplicitSdcCore)
    >>> # print the solution of the last iteration
    >>> print(my_solution.solution(-1).values)
    [[0.49999999999999989]
     [-1.1102230246251565e-16]]
    """
    def __init__(self, **kwargs):
        super(Sdc, self).__init__(**kwargs)
        self._num_time_steps = 1
        self.threshold = ThresholdCheck(min_threshold=1e-7, max_threshold=10, conditions=("residual", "iterations"))
        self.__exact = np.zeros(0)
        self._state = None
        self.__time_points = {
            'steps': np.zeros(0),
            'nodes': np.zeros(0)
        }
        self._deltas = {
            't': 0.0,
            'n': np.zeros(0)
        }
        self.timer = TimerBase()
        self._classic = True

    def init(self, problem, integrator=SdcIntegrator(), **kwargs):
        """Initializes SDC solver with given problem and integrator.

        Parameters
        ----------
        problem : :py:class:`.IInitialValueProblem`

        integrator : :py:class:`.SdcIntegrator`

        num_time_steps : :py:class:`int`
            Number of time steps to be used within the time interval of the problem.

        num_nodes : :py:class:`int`
            *(otional)*
            number of nodes per time step

        nodes_type : :py:class:`.INodes`
            *(optional)*
            Type of integration nodes to be used.

        weights_type : :py:class:`.IWeightFunction`
            *(optional)*
            Integration weights function to be used.

        classic : :py:class:`bool`
            *(optional)*
            Flag for specifying the type of the SDC sweep.

            :py:class:`True`
                *(default)*
                For the classic SDC as known from the literature
            :py:class:`False`
                For the modified SDC as developed by Torbjörn Klatt.

        Raises
        ------
        ValueError :

            * if given problem is not an :py:class:`.IInitialValueProblem`
            * if number of nodes per time step is not given; neither through ``num_nodes``, ``nodes_type`` nor
              ``integrator``

        See Also
        --------
        :py:meth:`.IIterativeTimeSolver.init` : overridden method
        """
        assert_is_instance(problem, IInitialValueProblem,
                           "SDC requires an initial value problem: {:s}".format(problem.__class__.__name__),
                           self)

        super(Sdc, self).init(problem, integrator, **kwargs)

        if 'num_time_steps' in kwargs:
            self._num_time_steps = kwargs['num_time_steps']

        if 'num_nodes' in kwargs:
            _num_nodes = kwargs['num_nodes']
        elif 'nodes_type' in kwargs and kwargs['nodes_type'].num_nodes is not None:
            _num_nodes = kwargs['nodes_type'].num_nodes
        elif integrator.nodes_type is not None and integrator.nodes_type.num_nodes is not None:
            _num_nodes = integrator.nodes_type.num_nodes
        else:
            raise ValueError(func_name(self) +
                             "Number of nodes per time step not given.")

        if 'nodes_type' not in kwargs:
            kwargs['nodes_type'] = GaussLobattoNodes()

        if 'weights_type' not in kwargs:
            kwargs['weights_type'] = PolynomialWeightFunction()

        if 'classic' in kwargs:
            assert_is_instance(kwargs['classic'], bool,
                               "Classic flag must either be True or False: NOT %s"
                               % kwargs['classic'].__class__.__name__,
                               self)
            self._classic = kwargs['classic']

        # initialize solver state
        self._state = SdcSolverState(num_nodes=_num_nodes - 1, num_time_steps=self.num_time_steps)

        # TODO: do we need this?
        _num_points = self.num_time_steps * (_num_nodes - 1) + 1

        self.__exact = np.zeros(_num_points, dtype=np.object)

        # compute time step and node distances
        self.state.delta_interval = self.problem.time_end - self.problem.time_start
        self._deltas['t'] = self.state.delta_interval / self.num_time_steps  # width of a single time step (equidistant)
        #  start time points of time steps
        self.__time_points['steps'] = np.linspace(self.problem.time_start,
                                                  self.problem.time_end, self.num_time_steps + 1)

        # initialize and transform integrator for time step width
        self._integrator.init(kwargs['nodes_type'], _num_nodes, kwargs['weights_type'],
                              interval=np.array([self.__time_points['steps'][0], self.__time_points['steps'][1]],
                                                dtype=np.float))
        del _num_nodes  # number of nodes is now always queried from integrator
        self.__time_points['nodes'] = np.zeros((self.num_time_steps, self.num_nodes), dtype=np.float)
        _deltas_n = np.zeros(self.num_time_steps * (self.num_nodes - 1) + 1)

        # copy the node provider so we do not alter the integrator's one
        _nodes = deepcopy(self._integrator.nodes_type)
        for _t in range(0, self.num_time_steps):
            # transform Nodes (copy) onto new time step for retrieving actual integration nodes
            _nodes.interval = \
                np.array([self.__time_points['steps'][_t], self.__time_points['steps'][_t + 1]])
            self.__time_points['nodes'][_t] = _nodes.nodes.copy()
            for _n in range(0, self.num_nodes - 1):
                _i = _t * (self.num_nodes - 1) + _n
                _deltas_n[_i + 1] = _nodes.nodes[_n + 1] - _nodes.nodes[_n]
        self._deltas['n'] = _deltas_n[1:].copy()

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

        See Also
        --------
        :py:meth:`.IIterativeTimeSolver.run` : overridden method
        """
        super(Sdc, self).run(core, **kwargs)

        # start logging output
        self._print_header()

        # initialize iteration timer of same type as global timer
        _iter_timer = self.timer.__class__()

        # start global timing
        self.timer.start()

        # start iterations
        self.__exact[0] = self.problem.initial_value

        # set initial values
        self.state.initial.solution.value = self.problem.initial_value
        self.state.initial.solution.time_point = self.problem.time_start
        self.state.initial.done()

        self._print_interval_header()

        while self.threshold.has_reached() is None:
            # initialize a new integration state
            self.state.proceed()

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
        # end while:self._threshold_check.has_reached() is None
        self.timer.stop()

        # finalize the IterativeSolution
        self.state.finalize()

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

        self._print_footer()

        return self.state.solution

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
        return self._integrator.nodes_type.num_nodes

    @property
    def classic(self):
        """Read-only accessor for the type of SDC

        Returns
        -------
        is_classic : :py:class:`bool`

            :py:class:`True`
                if it's the classic SDC as known from papers
            :py:class:`False`
                if it's the modified SDC by Torbjörn Klatt
        """
        return self._classic

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

        # for classic SDC compute integral
        _integral = 0.0
        _integrate_values = None
        if self.classic:
            _initial_value_rhs = self.problem.evaluate(self.state.current_time_step.initial.solution.time_point,
                                                       self.state.current_time_step.initial.solution.value)
            _integrate_values = np.array([_initial_value_rhs], dtype=self.problem.numeric_type)
            for _step_index in range(0, len(self.state.current_time_step)):
                if self.state.is_first_iteration:
                    _integrate_values = \
                        np.append(_integrate_values,
                                  np.array([_initial_value_rhs], dtype=self.problem.numeric_type),
                                  axis=0)
                else:
                    _step = self.state.previous_iteration[self.state.current_time_step_index][_step_index]
                    _integrate_values = \
                        np.append(_integrate_values,
                                  np.array([self.problem.evaluate(_step.solution.time_point, _step.solution.value)],
                                           dtype=self.problem.numeric_type),
                                  axis=0)
            del _initial_value_rhs

            assert_condition(_integrate_values.size == self.num_nodes,
                             ValueError, "Number of integration values not correct: {:d} != {:d}"
                                         .format(_integrate_values.size, self.num_nodes),
                             self)

        _full_integral = 0.0

        # do the actual SDC steps of this SDC sweep
        for _step_index in range(0, len(self.state.current_time_step)):
            _current_step = self.state.current_time_step[_step_index]
            if self.classic:
                _integral = self._integrator.evaluate(_integrate_values, last_node_index=_step_index + 1)
                _full_integral += _integral
            _current_step.integral = _integral.copy()
            # do the SDC step of this sweep
            self._sdc_step()
            if self.state.current_step_index < len(self.state.current_time_step) - 1:
                self.state.current_time_step.proceed()
        del _integrate_values

        # compute residual and print step details
        for _step_index in range(0, len(self.state.current_time_step)):
            _step = self.state.current_time_step[_step_index]

            self._core.compute_residual(self.state, step=_step, integral=_full_integral)

            # finalize this step (i.e. StepSolutionData.finalize())
            _step.done()

            if _step_index > 0:
                _previous_time = self.state.current_time_step[_step_index - 1].time_point
            else:
                _previous_time = self.state.current_time_step.initial.time_point

            if problem_has_exact_solution(self.problem, self):
                self._print_step(_step_index + 2,
                                 _previous_time,
                                 _step.time_point,
                                 supremum_norm(_step.solution.value),
                                 _step.solution.residual,
                                 _step.solution.error)
            else:
                self._print_step(_step_index + 2,
                                 _previous_time,
                                 _step.time_point,
                                 supremum_norm(_step.solution.value),
                                 _step.solution.residual,
                                 None)

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

        if not self.classic:
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
                    _this_value = self.problem.initial_value
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
            del _integrate_values
        # END if not self.classic

        # compute step
        self._core.run(self.state, problem=self.problem)

        # calculate error
        self._core.compute_error(self.state, problem=self.problem)

        # step gets finalized after computation of residual

    def _print_header(self):
        LOG.info("> " + '#' * 78)
        LOG.info("{:#<80}".format("> START: {:s} ".format(self._core.name)))
        LOG.info(">   Time Steps:             {:d}".format(self.num_time_steps))
        LOG.info(">   Integration Nodes:      {:d}".format(self.num_nodes))
        LOG.info(">   Termination Conditions: {:s}".format(self.threshold.print_conditions()))
        LOG.info(">   Problem:                {:s}".format(self.problem))
        LOG.info(">   Classic SDC:            {}".format(self.classic))

    def _print_interval_header(self):
        LOG.info(">   Interval:               [{:.3f}, {:.3f}]".format(self.problem.time_start, self.problem.time_end))
        self._print_output_tree_header()

    def _print_output_tree_header(self):
        LOG.info(">    iter")
        LOG.info(">         \\")
        LOG.info("!>          |- time    start     end        delta")
        LOG.info("!>          |     \\")
        LOG.info("!>          |      |- step    t_0      t_1       phi(t_1)    resid       err")
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
        LOG.info("{:#<80}".format("> FINISHED: {:s} ({:.3f} sec) ".format(self._core.name, self.timer.past())))
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
