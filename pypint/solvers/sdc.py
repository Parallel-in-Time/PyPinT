# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_iterative_time_solver import IIterativeTimeSolver
from copy import deepcopy
import warnings as warnings
import numpy as np
from .cores.sdc_core_mixin import SdcCoreMixin
from ..integrators.sdc_integrator import SdcIntegrator
from ..integrators.node_providers.gauss_lobatto_nodes import GaussLobattoNodes
from ..integrators.weight_function_providers.polynomial_weight_function import PolynomialWeightFunction
from ..problems import IInitialValueProblem, problem_has_exact_solution, problem_has_direct_implicit
from ..solutions.iterative_solution import IterativeSolution
from ..plugins.timers.timer_base import TimerBase
from ..utilities.threshold_check import ThresholdCheck
from ..utilities import assert_is_instance, assert_condition, func_name
from .. import LOG

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


class Sdc(IIterativeTimeSolver, SdcCoreMixin):
    """
    Summary
    -------
    *Spectral Deferred Corrections* method for solving first order ODEs.

    Extended Summary
    ----------------
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
    .IIterativeTimeSolver :
        implemented interface

    Examples
    --------
    >>> from pypint.solvers.sdc import Sdc
    >>> from examples.problems.constant import Constant
    >>> # setup the problem
    >>> my_problem = Constant(constant=-1.0)
    >>> # create the solver
    >>> my_solver = Sdc()
    >>> # initialize the solver with the problem
    >>> my_solver.init(problem=my_problem, num_time_steps=1, num_nodes=3)
    >>> # run the solver and get the solution
    >>> my_solution = my_solver.run()
    >>> # print the solution of the last iteration
    >>> print(my_solution.solution(-1))
    [  1.00000000e+00   5.00000000e-01  -1.11022302e-16]
    """

    class State(IIterativeTimeSolver.State):
        num_points = 0

        def __init__(self, iteration=0):
            super(Sdc.State, self).__init__(iteration)
            self._solution = np.zeros(Sdc.State.num_points)
            self._error = np.zeros(Sdc.State.num_points)
            self._residual = np.zeros(Sdc.State.num_points)
            self._reduction_of_solution = np.inf
            self._reduction_of_error = np.inf

        def solution_at(self, index, value=None):
            assert_condition(index < self.solution.size,
                             ValueError, "Index out of range: {:d} >= {:d}".format(index, self.solution.size),
                             self)
            if value is None:
                return self._solution[index]
            else:
                self._solution[index] = value

        def error_at(self, index, value=None):
            assert_condition(index < self.error.size,
                             ValueError, "Index out of range: {:d} >= {:d}".format(index, self.error.size),
                             self)
            if value is None:
                return self._error[index]
            else:
                self._error[index] = value

        def residual_at(self, index, value=None):
            assert_condition(index < self.residual.size,
                             ValueError, "Index out of range: {:d} >= {:d}".format(index, self.residual.size),
                             self)
            if value is None:
                return self._residual[index]
            else:
                self._residual[index] = value

    def __init__(self, **kwargs):
        super(Sdc, self).__init__(**kwargs)
        self.timer = TimerBase()
        self._num_time_steps = 1
        self._type = "expl"
        self._type_str = "Explicit"
        self.__num_nodes = 3
        self._threshold = ThresholdCheck(min_threshold=1e-7, max_threshold=10,
                                         conditions=("residual", "iterations"))
        self.__num_points = 0
        self.__exact = np.zeros(0)
        self.__time_points = {
            "steps": np.zeros(0),
            "nodes": np.zeros(0)
        }
        self._deltas = {
            "I": 0.0,
            "t": np.zeros(0),
            "n": np.zeros(0)
        }

    def init(self, problem, integrator=SdcIntegrator(), **kwargs):
        """
        Summary
        -------
        Initializes SDC solver with given problem and integrator.

        Parameters
        ----------
        In addition to the options available via :py:meth:`.IIterativeTimeSolver.init()`
        the following specific options are available:

        num_time_steps : integer
            Number of time steps to be used within the time interval of the problem.

        nodes_type : :py:class:`.INodes`
            Type of integration nodes to be used.

        weights_type : :py:class:`.IWeightFunction`
            Integration weights function to be used.

        type : str
            Specifying the type of the SDC steps being implicit (``impl``), explicit (``expl``) or semi-implicit
            (``semi``).
            Default is ``expl``.

        Raises
        ------
        ValueError
            If given problem is not an :py:class:`.IInitialValueProblem`.

        See Also
        --------
        .IIterativeTimeSolver.init
            overridden method
        """
        assert_is_instance(problem, IInitialValueProblem,
                           "SDC requires an initial value problem: {:s}".format(problem.__class__.__name__),
                           self)

        super(Sdc, self).init(problem, integrator, **kwargs)

        if "type" in kwargs and isinstance(kwargs["type"], str) and\
                (kwargs["type"] == "impl" or kwargs["type"] == "expl" or kwargs["type"] == "semi"):
            self._type = kwargs["type"]

        if self.is_implicit:
            self._type_str = "Implicit"
        elif self.is_explicit:
            self._type_str = "Explicit"
        else:
            self._type_str = "Semi-Implicit"

        if "num_time_steps" in kwargs:
            self._num_time_steps = kwargs["num_time_steps"]

        if "num_nodes" in kwargs:
            self.__num_nodes = kwargs["num_nodes"]
        elif "nodes_type" in kwargs and kwargs["nodes_type"].num_nodes is not None:
            self.__num_nodes = kwargs["nodes_type"].num_nodes
        elif integrator.nodes_type.num_nodes is not None:
            self.__num_nodes = integrator.nodes_type.num_nodes
        else:
            raise ValueError(func_name(self) +
                             "Number of nodes per time step not given.")

        if "nodes_type" not in kwargs:
            kwargs["nodes_type"] = GaussLobattoNodes()

        if "weights_type" not in kwargs:
            kwargs["weights_type"] = PolynomialWeightFunction()

        # initialize helper variables
        self.__num_points = self.num_time_steps * (self.__num_nodes - 1) + 1
        Sdc.State.num_points = self.__num_points

        # set the initial state
        self.states.append(Sdc.State())
        self.initial_state.solution = np.array([self.problem.initial_value] * self.__num_points,
                                               dtype=self.problem.numeric_type)
        self.initial_state.error = None
        self.initial_state.residual = None
        self.initial_state.reduction_of_solution = None
        self.initial_state.reduction_of_error = None

        self.__exact = np.zeros(self.__num_points, dtype=self.problem.numeric_type)

        # compute time step and node distances
        self._deltas["I"] = self.problem.time_end - self.problem.time_start
        _dt = self._deltas["I"] / self.num_time_steps
        self._deltas["t"] = np.array([_dt] * self.num_time_steps)
        self.__time_points["steps"] = np.linspace(self.problem.time_start,
                                                  self.problem.time_end, self.num_time_steps + 1)
        # initialize and transform integrator for time step width
        self._integrator.init(kwargs["nodes_type"], self.__num_nodes, kwargs["weights_type"],
                              interval=np.array([self.__time_points["steps"][0],
                                                 self.__time_points["steps"][1]]))
        self.__time_points["nodes"] = np.zeros(self.__num_points)
        self._deltas["n"] = np.zeros(self.num_time_steps * (self.num_nodes - 1) + 1)
        # copy the node provider so we do not alter the integrator's one
        _nodes = deepcopy(self._integrator.nodes_type)
        for _t in range(0, self.num_time_steps):
            # transform Nodes (copy) onto new time step for retrieving actual integration nodes
            _nodes.interval = \
                np.array([self.__time_points["steps"][_t], self.__time_points["steps"][_t + 1]])
            for _n in range(0, self.num_nodes - 1):
                _i = _t * (self.num_nodes - 1) + _n
                self.__time_points["nodes"][_i] = _nodes.nodes[_n]
                self._deltas["n"][_i + 1] = _nodes.nodes[_n + 1] - _nodes.nodes[_n]
        self.__time_points["nodes"][-1] = _nodes.nodes[-1]

        SdcCoreMixin.__init__(self)

    def run(self, solution_class=IterativeSolution):
        """
        Summary
        -------
        Applies SDC solver to the initialized problem setup.

        Extended Summary
        ----------------
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

            **sol**

                Computed solution for the time step.

            **resid**

                Residual of the time step.

            **err**

                Absolute error for the time step.
                Is only displayed if the given problem provides a function for the
                exact solution (see :py:meth:`.problem_has_exact_solution()`).

        Parameters
        ----------
        solution_class : class
            Class of the resulting solution.

        Returns
        -------
        solution : ISolution
            Solution object with solutions of each iteration.

        See Also
        --------
        .IIterativeTimeSolver.run
            overridden method
        """
        # init solution object
        _solution = solution_class(numeric_type=self.problem.numeric_type)

        # start logging output
        self._print_header()

        # initialize iteration timer of same type as global timer
        _iter_timer = self.timer.__class__()

        # start global timing
        self.timer.start()

        # start iterations
        self.__exact[0] = self.problem.initial_value
        _iter = 0
        while self.threshold.has_reached() is None:
            _iter += 1
            self.states.append(Sdc.State(iteration=_iter))
            self.current_state.solution = self.previous_state.solution.copy()

            # step result table header
            if problem_has_exact_solution(self.problem, self):
                self._output(["node", "t_0", "t_1", "sol", "resid", "err"],
                             ["str", "str", "str", "str", "str", "str"],
                             padding=10, debug=True)
            else:
                self._output(["node", "t_0", "t_1", "sol", "resid"],
                             ["str", "str", "str", "str", "str"],
                             padding=10, debug=True)

            # iterate on time steps
            _iter_timer.start()
            for self.core_state.time_step_index in range(0, self.num_time_steps):
                self._time_step()
            # end for:t
            _iter_timer.stop()

            # compute reduction
            if _iter > 1:
                # TODO: fix "overflow encountered in cdouble_scalars" and "invalid value encountered in cdouble_scalars"
                self.current_state.reduction_of_solution = \
                    np.abs((self.current_state.solution[-1] - self.current_state.solution[-1])
                           / self.previous_state.solution[-1] * 100.0)
                if problem_has_exact_solution(self.problem, self):
                    self.current_state.reduction_of_error = \
                        np.abs(self.current_state.error[-1] - self.current_state.error[-1])

            # log this iteration's summary
            if _iter == 1:
                # on first iteration we do not have comparison values
                self._output([_iter, None, _iter_timer.past()],
                             ["int", None, "float"],
                             padding=4)
            else:
                if problem_has_exact_solution(self.problem, self) and _iter > 0:
                    # we could compute the correct error of our current solution
                    self._output([_iter, self.current_state.reduction_of_solution, _iter_timer.past(),
                                  self.current_state.residual[-1], self.current_state.reduction_of_error],
                                 ["int", "exp", "float", "exp", "exp"],
                                 padding=4)
                else:
                    self._output([_iter, self.current_state.reduction_of_solution, _iter_timer.past(),
                                  self.current_state.residual],
                                 ["int", "exp", "float", "exp"],
                                 padding=4)

            # save solution for this iteration and check termination criteria
            if problem_has_exact_solution(self.problem, self):
                if _iter == 1:
                    _solution.add_solution(points=self.__time_points["nodes"],
                                           values=self.current_state.solution.copy(),
                                           exact=self.__exact,
                                           error=self.current_state.error.copy(),
                                           residual=self.current_state.residual.copy(),
                                           iteration=1)
                else:
                    _solution.add_solution(points=self.__time_points["nodes"],
                                           values=self.current_state.solution.copy(),
                                           error=self.current_state.error.copy(),
                                           residual=self.current_state.residual.copy(),
                                           iteration=-1)
                self.threshold.check(reduction=self.current_state.reduction_of_solution,
                                     residual=self.current_state.residual[-1],
                                     error=self.current_state.reduction_of_error,
                                     iterations=_iter)
            else:
                _solution.add_solution(points=self.__time_points["nodes"],
                                       values=self.current_state.solution,
                                       residual=self.current_state.residual,
                                       iteration=-1)
                self.threshold.check(reduction=self.current_state.reduction_of_solution,
                                     residual=self.current_state.residual[-1],
                                     iterations=_iter)
        # end while:self._threshold_check.has_reached() is None
        self.timer.stop()

        _solution.used_iterations = _iter

        if _iter <= self.threshold.max_iterations:
            LOG.info("> Converged after {:d} iteration(s).".format(_iter))
            LOG.info(">   {:s}".format(self.threshold.has_reached(human=True)))
            LOG.info(">   Rel. Reduction: {:.3e}".format(self.current_state.reduction_of_solution))
            LOG.info(">   Final Residual: {:.3e}".format(self.current_state.residual[-1]))
            if problem_has_exact_solution(self.problem, self):
                LOG.info(">   Absolute Error: {:.3e}".format(self.current_state.reduction_of_error))
        else:
            warnings.warn("Explicit SDC: Did not converged: {:s}".format(self.problem))
            LOG.info("> FAILED: After maximum of {:d} iteration(s).".format(_iter))
            LOG.info(">         Rel. Reduction: {:.3e}".format(self.current_state.reduction_of_solution))
            LOG.info(">         Final Residual: {:.3e}".format(self.current_state.residual[-1]))
            if problem_has_exact_solution(self.problem, self):
                LOG.info(">         Absolute Error: {:.3e}".format(self.current_state.reduction_of_error))
            LOG.warn("SDC Failed: Maximum number iterations reached without convergence.")

        _solution.reductions = {
            "solution": self.current_state.reduction_of_solution,
            "error": self.current_state.reduction_of_error
        }

        self._print_footer()

        return _solution

    @property
    def is_implicit(self):
        return self._type == "impl"

    @property
    def is_explicit(self):
        return self._type == "expl"

    @property
    def is_semi_implicit(self):
        return self._type == "semi"

    @property
    def num_time_steps(self):
        """
        Summary
        -------
        Accessor for the number of time steps within the interval.

        Returns
        -------
        number time steps : integer
            Number of time steps within the problem-given time interval.
        """
        return self._num_time_steps

    @property
    def num_nodes(self):
        """
        Summary
        -------
        Accessor for the number of integration nodes per time step.

        Returns
        -------
        number of nodes : integer
            Number of integration nodes used within one time step.
        """
        return self._integrator.nodes_type.num_nodes

    def _time_step(self):
        # transform integration nodes to next interval
        # _dT = self.__deltas["t"][t]
        # _T0 = self.__time_points["steps"][t]
        # _T1 = self.__time_points["steps"][t + 1]
        #LOG.debug("Time step {:d}: [{:2f}, {:.2f}] (dT={:2f}) with nodes: {:s}"
        #          .format(t + 1, _T0, _T1, _dT, self._integrator.nodes))
        for self.core_state.node_index in range(1, self.num_nodes):
            self._sdc_step()

    def _sdc_step(self):
        self.core_state.calculate_current_point_index()
        self.core_state.calculate_node_range()

        # get current steps' time data
        self.core_state.delta_tau = self._deltas["n"][self.core_state.node_index]
        self.core_state.current_time_point = self.__time_points["nodes"][self.core_state.previous_point_index]
        self.core_state.next_time_point = self.__time_points["nodes"][self.core_state.current_point_index]

        # gather values for integration
        _copy_mask = np.concatenate((np.array([True] * self.core_state.node_index),
                                     np.array([False] * (self.num_nodes - self.core_state.node_index))))
        _integrate_values = \
            np.where(_copy_mask,
                     self.current_state.solution[self.core_state.first_node_index:(self.core_state.last_node_index + 1)],
                     self.previous_state.solution[self.core_state.first_node_index:(self.core_state.last_node_index + 1)])

        # evaluate problem for integration values
        _integrate_values = \
            np.array([self.problem.evaluate(self._integrator.nodes[self.core_state.node_index - 1], val)
                      for val in _integrate_values], dtype=self.problem.numeric_type)

        # integrate
        _integral = self._integrator.evaluate(_integrate_values, last_node_index=self.core_state.node_index)

        # compute step
        self.execute_core(integral=_integral)

        # calculate residual
        _integrate_values = \
            np.where(_copy_mask,
                     self.current_state.solution[self.core_state.first_node_index:(self.core_state.last_node_index + 1)],
                     self.previous_state.solution[self.core_state.first_node_index:(self.core_state.last_node_index + 1)])
        _integrate_values[self.core_state.node_index] = self._states[-1].solution[self.core_state.current_point_index]
        _integrate_values = \
            np.array([self.problem.evaluate(self._integrator.nodes[self.core_state.node_index - 1], val)
                      for val in _integrate_values], dtype=self.problem.numeric_type)
        _residual_integral = 0
        for i in range(1, self.core_state.node_index + 1):
            _residual_integral += self._integrator.evaluate(_integrate_values, last_node_index=i)

        self.current_state.residual_at(self.core_state.current_point_index,
                                       np.abs(self.current_state.solution[self.core_state.first_node_index]
                                              + self._deltas["I"] * _residual_integral
                                              - self.current_state.solution[self.core_state.current_point_index]))

        # calculate error
        if problem_has_exact_solution(self.problem, self):
            self.__exact[self.core_state.current_point_index] = \
                self.problem.exact(self.core_state.next_time_point,
                                   self.__time_points["nodes"][self.core_state.current_point_index])
            self.current_state.error_at(self.core_state.current_point_index,
                                        np.abs(self.current_state.solution[self.core_state.current_point_index]
                                               - self.__exact[self.core_state.current_point_index]))
        else:
            # we need the exact solution for that
            #  (unless we find an error approximation method)
            pass

        # log
        if problem_has_exact_solution(self.problem, self):
            self._output([self.core_state.node_index, self.core_state.next_time_point,
                          self.core_state.current_time_point,
                          self.current_state.solution[self.core_state.current_point_index],
                          self.current_state.residual[self.core_state.current_point_index],
                          self.current_state.error[self.core_state.current_point_index]],
                         ["int", "float", "float", "float", "exp", "exp"],
                         padding=10, debug=True)
        else:
            self._output([self.core_state.node_index, self.core_state.next_time_point,
                          self.core_state.next_time_point,
                          self.current_state.solution[self.core_state.current_point_index],
                          self.current_state.residual[self.core_state.current_point_index]],
                         ["int", "float", "float", "float", "exp"],
                         padding=10, debug=True)

    def _print_header(self):
        LOG.info("> " + '#' * 78)
        LOG.info("{:#<80}".format("> START: {:s} SDC ".format(self._type_str)))
        LOG.info(">   Interval:               [{:.3f}, {:.3f}]".format(self.problem.time_start, self.problem.time_end))
        LOG.info(">   Time Steps:             {:d}".format(self.num_time_steps))
        LOG.info(">   Integration Nodes:      {:d}".format(self.num_nodes))
        LOG.info(">   Termination Conditions: {:s}".format(self.threshold.print_conditions()))
        LOG.info(">   Problem: {:s}".format(self.problem))
        LOG.info("> " + '-' * 78)

        # itartion result table header
        if problem_has_exact_solution(self.problem, self):
            self._output(["iter", "rel red", "time", "resid", "err red"],
                         ["str", "str", "str", "str", "str"],
                         padding=4)
        else:
            self._output(["iter", "rel red", "time", "resid"],
                         ["str", "str", "str", "str"],
                         padding=4)

    def _print_footer(self):
        LOG.info("{:#<80}".format("> FINISHED: {:s} SDC ({:.3f} sec) ".format(self._type_str, self.timer.past())))
        LOG.info("> " + '#' * 78)

    def _output(self, values, types, padding=0, debug=False):
        assert_condition(len(values) == len(types), ValueError, "Number of values must equal number of types.", self)
        _outstr = ' ' * padding
        for i in range(0, len(values)):
            if values[i] is None:
                _outstr += ' ' * 10
            else:
                if types[i] == "float":
                    _outstr += "{: 10.3f}".format(values[i])
                elif types[i] == "int":
                    _outstr += "{: 10d}".format(values[i])
                elif types[i] == "exp":
                    _outstr += "{: 10.2e}".format(values[i])
                elif types[i] == "str":
                    _outstr += "{: >10s}".format(values[i])
                else:
                    raise ValueError(func_name(self) +
                                     "Given type for value '{:s}' is invalid: {:s}"
                                     .format(values[i], types[i]))
            _outstr += "    "
        if debug:
            LOG.debug("!> {:s}".format(_outstr))
        else:
            LOG.info("> {:s}".format(_outstr))
