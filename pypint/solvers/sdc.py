# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_iterative_time_solver import IIterativeTimeSolver
from copy import deepcopy
import warnings as warnings
import numpy as np
from math import fabs
from pypint.integrators.sdc_integrator import SdcIntegrator
from pypint.integrators.node_providers.gauss_lobatto_nodes \
    import GaussLobattoNodes
from pypint.integrators.weight_function_providers.polynomial_weight_function \
    import PolynomialWeightFunction
from pypint.problems.i_initial_value_problem import IInitialValueProblem
from pypint.solutions.iterative_solution import IterativeSolution
from pypint.plugins.timers.timer_base import TimerBase
from pypint.utilities.threshold_check import ThresholdCheck
from pypint.utilities import func_name
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

    Notes
    -----
    Currently, only the explicit version is implemented.

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

    .. todo:: Implement implicit SDC method for first order ODEs.
    """
    def __init__(self, **kwargs):
        super(Sdc, self).__init__(**kwargs)
        self.timer = TimerBase()
        self._num_time_steps = 1
        self._implicit = False
        self.__num_nodes = 3
        self._threshold = ThresholdCheck(min_threshold=1e-7, max_threshold=10,
                                         conditions=("residual", "iterations"))
        self.__num_points = 0
        self.__sol = {
            "previous": np.zeros(0),
            "current": np.zeros(0)
        }
        self.__exact = np.zeros(0)
        self.__time_points = {
            "steps": np.zeros(0),
            "nodes": np.zeros(0)
        }
        self.__deltas = {
            "I": 0.0,
            "t": np.zeros(0),
            "n": np.zeros(0)
        }
        # absolute errors
        self.__errors = {
            "previous": np.zeros(0),
            "current": np.zeros(0)
        }
        self.__reductions = {
            "solution": np.zeros(0),
            "errors": np.zeros(0)
        }
        # residuals
        self.__residuals = {
            "previous": np.zeros(0),
            "current": np.zeros(0)
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

        Raises
        ------
        ValueError
            If given problem is not an :py:class:`.IInitialValueProblem`.

        See Also
        --------
        .IIterativeTimeSolver.init
            overridden method
        """
        if not isinstance(problem, IInitialValueProblem):
            raise ValueError(func_name(self) +
                             "SDC requires an initial value.")

        super(Sdc, self).init(problem, integrator, **kwargs)

        if "implicit" in kwargs and isinstance(kwargs["implicit"], bool):
            self._implicit = kwargs["implicit"]

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
        self.__sol["current"] = np.array([self.problem.initial_value] * self.__num_points)
        self.__sol["previous"] = self.__sol["current"].copy()
        self.__exact = np.zeros(self.__num_points)
        self.__errors["current"] = np.array([0.0] * self.__num_points)
        self.__errors["previous"] = self.__errors["current"].copy()
        self.__reductions["solution"] = np.ones(self.threshold.max_iterations + 1)
        self.__reductions["errors"] = np.ones(self.threshold.max_iterations + 1)
        self.__residuals["current"] = np.array([0.0] * self.__num_points)
        self.__residuals["previous"] = self.__residuals["current"].copy()

        # compute time step and node distances
        self.__deltas["I"] = self.problem.time_end - self.problem.time_start
        _dt = self.__deltas["I"] / self.num_time_steps
        self.__deltas["t"] = np.array([_dt] * self.num_time_steps)
        self.__time_points["steps"] = np.linspace(self.problem.time_start,
                                                  self.problem.time_end, self.num_time_steps + 1)
        # initialize and transform integrator for time step width
        self._integrator.init(kwargs["nodes_type"], self.__num_nodes, kwargs["weights_type"],
                              interval=np.array([self.__time_points["steps"][0],
                                                 self.__time_points["steps"][1]]))
        self.__time_points["nodes"] = np.zeros(self.__num_points)
        self.__deltas["n"] = np.zeros(self.num_time_steps * (self.num_nodes - 1) + 1)
        # copy the node provider so we do not alter the integrator's one
        _nodes = deepcopy(self._integrator.nodes_type)
        for _t in range(0, self.num_time_steps):
            # transform Nodes (copy) onto new time step for retrieving actual integration nodes
            _nodes.interval = \
                np.array([self.__time_points["steps"][_t], self.__time_points["steps"][_t + 1]])
            for _n in range(0, self.num_nodes - 1):
                _i = _t * (self.num_nodes - 1) + _n
                self.__time_points["nodes"][_i] = _nodes.nodes[_n]
                #LOG.debug("    i={:d}*{:d}+{:d}={:d}".format(_t, (self.num_nodes-1), _n, _i))
                self.__deltas["n"][_i + 1] = _nodes.nodes[_n + 1] - _nodes.nodes[_n]
                #LOG.debug("      dist([{:.2f}, {:.2f}]) = {:.2f}"
                #          .format(self._integrator.nodes[_n], self.
                #                  integrator.nodes[_n + 1], self.__deltas["n"][_i + 1]))
        self.__time_points["nodes"][-1] = _nodes.nodes[-1]
        #LOG.debug("self.__deltas['n']: {:s}".format(self.__deltas["n"]))

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
                provides a function of the exact solution (see :py:meth:`.IProblem.has_exact()`).

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
                exact solution (see :py:meth:`.IProblem.has_exact()`).

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
        _sol = solution_class()

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

            # step result table header
            if self.problem.has_exact():
                self._output(["node", "t_0", "t_1", "sol", "resid", "err"],
                             ["str", "str", "str", "str", "str", "str"],
                             padding=10, debug=True)
            else:
                self._output(["node", "t_0", "t_1", "sol", "resid"],
                             ["str", "str", "str", "str", "str"],
                             padding=10, debug=True)

            # iterate on time steps
            _iter_timer.start()
            for t in range(0, self.num_time_steps):
                self._time_step(t)
            # end for:t
            _iter_timer.stop()

            # compute reduction
            if _iter > 1:
                self.__reductions["solution"][_iter - 1] = \
                    fabs((self.__sol["previous"][-1] - self.__sol["current"][-1])
                         / self.__sol["previous"][-1] * 100.0)
                if self.problem.has_exact():
                    self.__reductions["errors"][_iter - 1] = \
                        fabs(self.__errors["previous"][-1] - self.__errors["current"][-1])

            # log this iteration's summary
            if _iter == 1:
                # on first iteration we do not have comparison values
                self._output([_iter, None, _iter_timer.past()],
                             ["int", None, "float"],
                             padding=4)
            else:
                if self.problem.has_exact() and _iter > 0:
                    # we could compute the correct error of our current solution
                    self._output([_iter, self.__reductions["solution"][_iter - 1],
                                  _iter_timer.past(), self.__residuals["current"][-1],
                                  self.__reductions["errors"][_iter - 1]],
                                 ["int", "exp", "float", "exp", "exp"],
                                 padding=4)
                else:
                    self._output([_iter, self.__reductions["solution"][_iter - 1],
                                  _iter_timer.past(), self.__residuals["current"][-1]],
                                 ["int", "exp", "float", "exp"],
                                 padding=4)

            # save solution for this iteration and check termination criteria
            if self.problem.has_exact():
                if _iter == 1:
                    _sol.add_solution(points=self.__time_points["nodes"],
                                      values=self.__sol["current"].copy(),
                                      exact=self.__exact,
                                      error=self.__errors["current"].copy(),
                                      residual=self.__residuals["current"].copy(),
                                      iteration=-1)
                else:
                    _sol.add_solution(points=self.__time_points["nodes"],
                                      values=self.__sol["current"].copy(),
                                      error=self.__errors["current"].copy(),
                                      residual=self.__residuals["current"].copy(),
                                      iteration=-1)
                self.threshold.check(reduction=self.__reductions["solution"][_iter - 2],
                                     residual=self.__residuals["current"][-1],
                                     error=self.__reductions["errors"][_iter - 2],
                                     iterations=_iter)
            else:
                _sol.add_solution(points=self.__time_points["nodes"],
                                  values=self.__sol["current"].copy(),
                                  residual=self.__residuals["current"].copy(),
                                  iteration=-1)
                self.threshold.check(reduction=self.__reductions["solution"][_iter - 2],
                                     residual=self.__residuals["current"][-1],
                                     iterations=_iter)

            # reset helper variables
            self.__sol["previous"] = self.__sol["current"].copy()
            self.__residuals["previous"] = self.__residuals["current"].copy()
            self.__residuals["current"] = np.zeros(self.__residuals["current"].size)
            if self.problem.has_exact():
                self.__errors["previous"] = self.__errors["current"].copy()
                self.__errors["current"] = np.zeros(self.__errors["current"].size)
        # end while:self._threshold_check.has_reached() is None
        self.timer.stop()

        if _iter <= self.threshold.max_iterations:
            LOG.info("> Converged after {:d} iteration(s).".format(_iter))
            LOG.info(">   {:s}".format(self.threshold.has_reached(human=True)))
            LOG.info(">   Rel. Reduction: {:.3e}".format(self.__reductions["solution"][_iter - 1]))
            LOG.info(">   Final Residual: {:.3e}".format(self.__residuals["previous"][-1]))
            if self.problem.has_exact():
                LOG.info(">   Absolute Error: {:.3e}"
                         .format(self.__reductions["errors"][_iter - 1]))
        else:
            warnings.warn("Explicit SDC: Did not converged!")
            LOG.info("> FAILED: After maximum of {:d} iteration(s).".format(_iter))
            LOG.info(">         Rel. Reduction: {:.3e}"
                     .format(self.__reductions["solution"][_iter - 1]))
            LOG.info(">         Final Residual: {:.3e}"
                     .format(self.__residuals["previous"][-1]))
            if self.problem.has_exact():
                LOG.info(">         Absolute Error: {:.3e}"
                         .format(self.__reductions["errors"][_iter - 1]))
            LOG.warn("SDC Failed: Maximum number iterations reached without convergence.")

        _sol.reductions = self.__reductions

        self._print_footer()

        return _sol

    @property
    def is_implicit(self):
        return self._implicit

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

    def _time_step(self, t):
        # transform integration nodes to next interval
        _dT = self.__deltas["t"][t]
        _T0 = self.__time_points["steps"][t]
        _T1 = self.__time_points["steps"][t + 1]
        #LOG.debug("Time step {:d}: [{:2f}, {:.2f}] (dT={:2f}) with nodes: {:s}"
        #          .format(t + 1, _T0, _T1, _dT, self._integrator.nodes))
        for n in range(1, self.num_nodes):
            self._sdc_step(t, n)

    def _sdc_step(self, t, n):
        _i = t * (self.num_nodes - 1) + n
        _i0 = _i - n
        _i1 = (t + 1) * (self.num_nodes - 1)

        # get current steps' time data
        _dt = self.__deltas["n"][n]
        _t0 = self.__time_points["nodes"][_i - 1]
        _t1 = self.__time_points["nodes"][_i]

        #LOG.debug("integration step {:d}: [{:2f}, {:.2f}] (dT={:2f})".format(n, _t0, _t1, _dt))

        # gather values for integration
        _copy_mask = np.concatenate((np.array([True] * n),
                                     np.array([False] * (self.num_nodes - n))))
        _integrate_values = np.where(_copy_mask, self.__sol["current"][_i0:(_i1 + 1)],
                                     self.__sol["previous"][_i0:(_i1 + 1)])

        # evaluate problem for integration values
        _integrate_values = \
            np.array([self.problem.evaluate(self._integrator.nodes[n - 1], val)
                      for val in _integrate_values])

        # integrate
        integral = self._integrator.evaluate(_integrate_values, last_node_index=n)

        # compute step
        if self.is_implicit:
            _expl_term = self.__sol["current"][_i - 1] - _dt * self.problem.evaluate(_t0, self.__sol["previous"][_i]) + self.__deltas["I"] * integral
            _func = lambda x_next: _expl_term + _dt * self.problem.evaluate(_t0, x_next[0]) - x_next
            self.__sol["current"][_i] = self.problem.implicit_solve(np.array([self.__sol["current"][_i]]), _func)
        else:
            self.__sol["current"][_i] = \
                self.__sol["current"][_i - 1] + \
                _dt * (self.problem.evaluate(_t0, self.__sol["current"][_i - 1]) -
                       self.problem.evaluate(_t0, self.__sol["previous"][_i - 1])) + \
                self.__deltas["I"] * integral
            #LOG.debug("          {:f} = {:f} + {:f} * ({:f} - {:f}) + {:f} * {:f}"
            #          .format(self.__sol["current"][_i], self.__sol["current"][_i - 1], _dt,
            #                  self.problem.evaluate(_t0, self.__sol["current"][_i - 1]),
            #                  self.problem.evaluate(_t0, self.__sol["previous"][_i - 1]),
            #                  self.__deltas["I"], integral))

        # calculate residual
        _integrate_values = np.where(_copy_mask,
                                     self.__sol["current"][_i0:(_i1 + 1)],
                                     self.__sol["previous"][_i0:(_i1 + 1)])
        _integrate_values[n] = self.__sol["current"][_i]
        _integrate_values = \
            np.array([self.problem.evaluate(self._integrator.nodes[n - 1], val)
                      for val in _integrate_values])
        _residual_integral = 0
        for i in range(1, n + 1):
            _residual_integral += self._integrator.evaluate(_integrate_values, last_node_index=i)

        self.__residuals["current"][_i] = \
            fabs(self.__sol["current"][_i0] + self.__deltas["I"] * _residual_integral
                 - self.__sol["current"][_i])
        #LOG.debug("          Residual: {:f} = abs({:f} + {:f} * {:f} - {:f})"
        #          .format(self.__residuals["current"][_i], self.__sol["current"][_i0],
        #                  self.__deltas["I"], _residual_integral, self.__sol["current"][_i]))

        # calculate error
        if self.problem.has_exact():
            self.__exact[_i] = self.problem.exact(_t0, self.__time_points["nodes"][_i])
            self.__errors["current"][_i] = fabs(self.__sol["current"][_i] - self.__exact[_i])
        else:
            # we need the exact solution for that
            #  (unless we find an error approximation method)
            pass

        # log
        if self.problem.has_exact():
            self._output([_i, _t0, _t1, self.__sol["current"][_i], self.__residuals["current"][_i],
                          self.__errors["current"][_i]],
                         ["int", "float", "float", "float", "exp", "exp"],
                         padding=10, debug=True)
        else:
            self._output([_i, _t0, _t1, self.__sol["current"][_i], self.__residuals["current"][_i]],
                         ["int", "float", "float", "float", "exp"],
                         padding=10, debug=True)

    def _print_header(self):
        LOG.info("> " + '#' * 78)
        LOG.info("{:#<80}".format("> START: {:s} SDC ".format("Implicit" if self.is_implicit else "Explicit")))
        LOG.info(">   Interval:               [{:.3f}, {:.3f}]"
                 .format(self.problem.time_start, self.problem.time_end))
        LOG.info(">   Time Steps:             {:d}".format(self.num_time_steps))
        LOG.info(">   Integration Nodes:      {:d}".format(self.num_nodes))
        LOG.info(">   Termination Conditions: {:s}".format(self.threshold.print_conditions()))
        LOG.info(">   Problem: {:s}".format(self.problem))
        LOG.info("> " + '-' * 78)

        # itartion result table header
        if self.problem.has_exact():
            self._output(["iter", "rel red", "time", "resid", "err red"],
                         ["str", "str", "str", "str", "str"],
                         padding=4)
        else:
            self._output(["iter", "rel red", "time", "resid"],
                         ["str", "str", "str", "str"],
                         padding=4)

    def _print_footer(self):
        LOG.info("{:#<80}".format("> FINISHED: Explicit SDC ({:.3f} sec) "
                                  .format(self.timer.past())))
        LOG.info("> " + '#' * 78)

    def _output(self, values, types, padding=0, debug=False):
        if len(values) != len(types):
            raise ValueError(func_name(self) +
                             "Number of values must equal number of types.")
        outstr = ' ' * padding
        for i in range(0, len(values)):
            if values[i] is None:
                outstr += ' ' * 10
            else:
                if types[i] == "float":
                    outstr += "{: 10.3f}".format(values[i])
                elif types[i] == "int":
                    outstr += "{: 10d}".format(values[i])
                elif types[i] == "exp":
                    outstr += "{: 10.2e}".format(values[i])
                elif types[i] == "str":
                    outstr += "{: >10s}".format(values[i])
                else:
                    raise ValueError(func_name(self) +
                                     "Given type for value '{:s}' is invalid: {:s}"
                                     .format(values[i], types[i]))
            outstr += "    "
        if debug:
            LOG.debug("!> {:s}".format(outstr))
        else:
            LOG.info("> {:s}".format(outstr))
