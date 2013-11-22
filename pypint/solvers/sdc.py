# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_iterative_time_solver import IIterativeTimeSolver
import warnings as warnings
import numpy as np
from pypint.integrators.sdc_integrator import SdcIntegrator
from pypint.integrators.node_providers.gauss_lobatto_nodes \
    import GaussLobattoNodes
from pypint.integrators.weight_function_providers.polynomial_weight_function \
    import PolynomialWeightFunction
from pypint.problems.i_initial_value_problem import IInitialValueProblem
from pypint.solutions.iterative_solution import IterativeSolution
from pypint.plugins.timers.timer_base import TimerBase
from pypint.utilities import func_name
from pypint import LOG


class Sdc(IIterativeTimeSolver):
    """
    Summary
    -------
    *Spectral Deferred Corrections* method for solving first order ODEs.

    Extended Summary
    ----------------
    The *Spectral Deferred Corrections* (SDC) method is described in [Minion2003]_ (Equation 2.7)

    Default Values:

        :py:attr:`.max_iterations`: 5

        :py:attr:`.min_reduction`: 1e-7

        :py:attr:`.num_time_steps`: 2

    Notes
    -----
    Currently, only the explicit version is implemented.

    .. todo:: Implement implicit SDC method for first order ODEs.
    """
    def __init__(self):
        super(self.__class__, self).__init__()
        self.timer = TimerBase()
        self.max_iterations = 5
        self.min_reduction = 1e-7
        self._num_time_steps = 2
        self.__sol = {
            "previous": np.zeros(0),
            "current": np.zeros(0)
        }
        self.__delta_times = {
            "steps": np.zeros(0),
            "interval": 0
        }

    def init(self, problem, integrator=SdcIntegrator(), **kwargs):
        """
        Summary
        -------
        Initializes SDC solver with given problem and integrator.

        Parameters
        ----------
        kwargs : further named arguments
            In addition to the options available via :py:meth:`.IIterativeTimeSolver.init()`
            the following specific options are available:

            ``num_time_steps`` : integer
                Number of time steps to be used within the time interval of the problem.

            ``nodes_type`` : :py:class:`.INodes`
                Type of integration nodes to be used.

            ``weights_type`` : :py:class:`.IWeightFunction`
                Integration weights function to be used.

        See Also
        --------
        .IIterativeTimeSolver.init
            overridden method
        """
        if not isinstance(problem, IInitialValueProblem):
            raise ValueError(func_name(self) +
                             "SDC requires an initial value.")
        super(self.__class__, self).init(problem, integrator)
        if "num_time_steps" not in kwargs:
            kwargs["num_time_steps"] = self.num_time_steps
        if "nodes_type" not in kwargs:
            kwargs["nodes_type"] = GaussLobattoNodes()
        if "weights_type" not in kwargs:
            kwargs["weights_type"] = PolynomialWeightFunction()

        self._num_time_steps = kwargs["num_time_steps"]

        # initialize integrator
        self._integrator.init(kwargs["nodes_type"], self.num_time_steps + 1,
                              kwargs["weights_type"],
                              np.array([self.problem.time_start, self.problem.time_end]))

        # initialize helper variables
        self.__sol["previous"] = np.array([self.problem.initial_value] * (self.num_time_steps + 1))
        self.__sol["current"] = np.zeros(self.num_time_steps + 1)

        # compute time step distances
        self.__delta_times["interval"] = self.problem.time_end - self.problem.time_start
        self.__delta_times["steps"] = \
            np.array([self._integrator.nodes[i+1] - self._integrator.nodes[i]
                      for i in range(0, self.num_time_steps)])

    def run(self):
        """
        Summary
        -------
        Applies SDC solver to the initialized problem setup.

        Extended Summary
        ----------------
        Solves the given problem with the explicit SDC algorithm.

        Returns
        -------
        solution : IterativeSolution
            Solution object with solutions of each iteration.
        """
        # init solution object
        _sol = IterativeSolution()

        # start logging output
        LOG.info('#' * 80)
        LOG.info("{:#<80}".format("# START: Explicit SDC "))
        LOG.info("#  Interval:       [{:f}, {:f}]".format(self.problem.time_start,
                                                          self.problem.time_end))
        LOG.info("#  Time Steps:     {:d}".format(self.num_time_steps))
        LOG.info("#  Max Iterations: {:d}".format(self.max_iterations))
        LOG.info("#  Min Reduction:  {:e}".format(self.min_reduction))
        LOG.info('-' * 80)

        # itartion result table header
        LOG.info(' ' * 4 + "{: >6s}    {: >8s}    {: >8s}".format("iter", "rel red", "time"))

        # initialize iteration timer of same type as global timer
        _iter_timer = self.timer.__class__()

        # start global timing
        self.timer.start()

        # start iterations
        _relred = 99.0  # dummy value
        _iter = 0
        while _relred > self.min_reduction:
            _iter_timer.start()

            # iterate on time steps
            for step in range(0, self.num_time_steps):
                self._sdc_step(step)
            # end for:step

            # compute reduction
            _relred = (self.__sol["previous"][-1] - self.__sol["current"][-1]) / self.__sol["previous"][-1] * 100.0

            _iter_timer.stop()

            # log this iteration's summary
            if _iter == 0:
                LOG.info(' ' * 4 + "{: 6d}    {:s}    {: 8.4f}"
                                   .format(1, ' ' * 8, _iter_timer.past()))
            else:
                LOG.info(' ' * 4 + "{: 6d}    {: 8.4f}    {: 8.4f}"
                                   .format(_iter + 1, _relred, _iter_timer.past()))

            # save solution for this iteration
            _sol.add_solution(_iter, self.__sol["current"])
            self.__sol["previous"] = self.__sol["current"].copy()
            self.__sol["current"] = np.zeros(self.num_time_steps + 1)

            _iter += 1

            # check maximum iterations
            if _iter == self.max_iterations:
                # and stop iterating if reached
                break
        # end while:_relred
        self.timer.stop()

        if _iter <= self.max_iterations and _relred <= self.min_reduction:
            LOG.info("# Converged after {:d} iteration(s): {:f}".format(_iter, _relred))
        else:
            warnings.warn("Explicit SDC: Max iterations reached without convergence.")
            LOG.info("# FAILED: relative reduction after {:d} iteration(s): {:f}"
                     .format(_iter + 1, _relred))
            LOG.warn("SDC Failed: Maximum number iterations reached without convergence.")

        LOG.info("{:#<80}".format("# FINISHED: Explicit SDC ({:.3f} sec) "
                                  .format(self.timer.past())))
        LOG.info('#' * 80)
        return _sol

    @property
    def num_time_steps(self):
        """
        Summary
        -------
        Accessor for the number of time steps within the interval

        Returns
        -------
        number time steps : integer
            Number of intermediate time steps within the problem-given time interval.
        """
        return self._num_time_steps

    def _sdc_step(self, step):
        # get current steps' time data
        _dt = self.__delta_times["steps"][step]
        _time = self._integrator.nodes[step]

        #LOG.debug(' ' * 4 + "Time Step {: 2d}:".format(step + 1))
        #LOG.debug(' ' * 6 + "Interval: [{:f}, {:f}], dt: {:f}"
                            #.format(step + 1, _time, self._integrator.nodes[step + 1], _dt))

        # gather values for integration
        _copy_mask = np.concatenate((np.array([True] * step),
                                     np.array([False] * (self.num_time_steps - step + 1))))
        _integrate_values = np.where(_copy_mask, self.__sol["current"],
                                     self.__sol["previous"])

        # evaluate problem for integration values
        _integrate_values = \
            np.array([self.problem.evaluate(self._integrator.nodes[step], val)
                      for val in _integrate_values])

        # integrate
        integral = self._integrator.evaluate(_integrate_values, until_node_index=step)

        # compute step
        self.__sol["current"][step + 1] = \
            self.__sol["current"][step] + \
            _dt * (self.problem.evaluate(_time, self.__sol["current"][step]) -
                   self.problem.evaluate(_time, self.__sol["previous"][step])) + \
            self.__delta_times["interval"] * integral
