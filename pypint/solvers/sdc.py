# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_iterative_time_solver import IIterativeTimeSolver
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
    def __init__(self, **kwargs):
        super(Sdc, self).__init__(**kwargs)
        self.timer = TimerBase()
        self._num_time_steps = 2
        self.__sol = {
            "previous": np.zeros(0),
            "current": np.zeros(0)
        }
        self.__delta_times = {
            "steps": np.zeros(0),
            "interval": 0
        }
        # absolute errors
        self.__err_vec = {
            "previous": np.zeros(0),
            "current": np.zeros(0)
        }
        self.__err_red_rate_vec = np.zeros(0)

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
        super(Sdc, self).init(problem, integrator, **kwargs)
        if self.max_iterations is None:
            self.max_iterations = 5

        if self.min_reduction is None:
            self.min_reduction = 1e-7

        if "num_time_steps" in kwargs:
            self._num_time_steps = kwargs["num_time_steps"]
        else:
            self._num_time_steps = 2

        if "nodes_type" not in kwargs:
            kwargs["nodes_type"] = GaussLobattoNodes()
        if "weights_type" not in kwargs:
            kwargs["weights_type"] = PolynomialWeightFunction()

        # initialize integrator
        self._integrator.init(kwargs["nodes_type"], self.num_time_steps + 1,
                              kwargs["weights_type"],
                              np.array([self.problem.time_start, self.problem.time_end]))

        # initialize helper variables
        self.__sol["previous"] = np.array([self.problem.initial_value] * (self.num_time_steps + 1))
        self.__sol["current"] = self.__sol["previous"].copy()
        self.__err_vec["previous"] = np.array([0.0] * self.num_time_steps)
        self.__err_vec["current"] = self.__err_vec["previous"].copy()

        # compute time step distances
        self.__delta_times["interval"] = self.problem.time_end - self.problem.time_start
        self.__delta_times["steps"] = \
            np.array([self._integrator.nodes[i+1] - self._integrator.nodes[i]
                      for i in range(0, self.num_time_steps)])

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

            **err red**

                Reduction of the absolute error from the first iteration to the current.
                Is only displayed from the second iteration onwards and only if the given problem
                provides a function of the exact solution (see :py:meth:`.IProblem.has_exact()`).

        The output for the time steps of an iteration explained:

            **step**

                Number of the time step.

            **t_0**

                Start of the time step interval.

            **t_1**

                End of the time step interval.

            **sol**

                Computed solution for the time step.

            **err**

                Absolute error for the time step.
                Is only displayed if the given problem provides a function for the
                exact solution (see :py:meth:`.IProblem.has_exact()`).

        Returns
        -------
        solution : ISolution
            Solution object with solutions of each iteration.
        """

        # init solution object
        _sol = solution_class()

        # start logging output
        LOG.info('#' * 80)
        LOG.info("{:#<80}".format("# START: Explicit SDC "))
        LOG.info("#  Interval:       [{:.3f}, {:.3f}]".format(self.problem.time_start,
                                                              self.problem.time_end))
        LOG.info("#  Time Steps:     {:d}".format(self.num_time_steps))
        LOG.info("#  Max Iterations: {:d}".format(self.max_iterations))
        LOG.info("#  Min Reduction:  {:.2e}".format(self.min_reduction))
        LOG.info("#  Problem: {:s}".format(self.problem))
        LOG.info('-' * 80)

        # itartion result table header
        if self.problem.has_exact():
            LOG.info(' ' * 4 + "{: >4s}    {: >10s}    {: >8s}    {: >10s}"
                               .format("iter", "rel red", "time", "err red"))
        else:
            LOG.info(' ' * 4 + "{: >4s}    {: >10s}    {: >8s}"
                               .format("iter", "rel red", "time"))

        # initialize iteration timer of same type as global timer
        _iter_timer = self.timer.__class__()

        # start global timing
        self.timer.start()

        # start iterations
        _relred = 1.0
        _errred = 1.0
        _iter = -1
        _converged = False
        while not _converged:
            _iter += 1

            # step result table header
            if self.problem.has_exact():
                LOG.debug(' ' * 10 + "{: >4s}    {: >6s}    {: >6s}    {: >10s}    {: >10s}"
                                     .format("step", "t_0", "t_1", "sol", "err"))
            else:
                LOG.debug(' ' * 10 + "{: >4s}    {: >6s}    {: >6s}    {: >10s}"
                                     .format("step", "t_0", "t_1", "sol"))

            # iterate on time steps
            _iter_timer.start()
            for step in range(0, self.num_time_steps):
                # get current steps' time data
                _dt = self.__delta_times["steps"][step]
                _time = self._integrator.nodes[step]

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
                #LOG.debug("          {:f} = {:f} + {:f} * ({:f} - {:f}) + {:f} * {:f}"
                          #.format(self.__sol["current"][step + 1], self.__sol["current"][step], _dt,
                          #        self.problem.evaluate(_time, self.__sol["current"][step]),
                          #        self.problem.evaluate(_time, self.__sol["previous"][step]),
                          #        self.__delta_times["interval"], integral))

                # calculate error and its reduction
                if self.problem.has_exact():
                    self.__err_vec["current"][step] = \
                        fabs(self.__sol["current"][step+1] -
                             self.problem.exact(_time, self._integrator.nodes[step + 1]))
                else:
                    # we need the exact solution for that
                    #  (unless we find an error approximation method)
                    pass

                # log
                if self.problem.has_exact():
                    LOG.debug(' ' * 10 + "{: >4d}    {: 6.2f}    {: 6.2f}    {: 10.4f}    {: 10.2e}"
                                         .format(step+1, _time, self._integrator.nodes[step+1],
                                                 self.__sol["current"][step+1],
                                                 self.__err_vec["current"][step]))
                else:
                    LOG.debug(' ' * 10 + "{: >4d}    {: 6.2f}    {: 6.2f}    {: 10.4f}"
                                         .format(step+1, _time, self._integrator.nodes[step+1],
                                                 self.__sol["current"][step+1]))
            # end for:step
            _iter_timer.stop()

            # compute reduction
            _relred = fabs((self.__sol["previous"][-1] - self.__sol["current"][-1])
                           / self.__sol["previous"][-1] * 100.0)
            if self.problem.has_exact():
                _errred = fabs(self.__err_vec["previous"][-1] - self.__err_vec["current"][-1])

            # log this iteration's summary
            if _iter == 0:
                # on first iteration we do not have comparison values
                LOG.info(' ' * 4 + "{: 4d}    {:s}    {: 8.4f}"
                                   .format(1, ' ' * 10, _iter_timer.past()))
            else:
                if self.problem.has_exact() and _iter > 0:
                    # we could compute the correct error of our current solution
                    LOG.info(' ' * 4 + "{: 4d}    {: 10.2e}    {: 8.4f}    {: 10.2e}"
                                       .format(_iter + 1, _relred, _iter_timer.past(), _errred))
                else:
                    LOG.info(' ' * 4 + "{: 4d}    {: 10.2e}    {: 8.4f}"
                                       .format(_iter + 1, _relred, _iter_timer.past()))

            # save solution for this iteration
            _sol.add_solution(_iter, self.__sol["current"])

            # update converged flag
            _converged = _converged or _relred <= self.min_reduction
            if self.problem.has_exact:
                _converged = _converged or _errred <= self.min_reduction

            # check maximum iterations
            if _iter == self.max_iterations:
                # and stop iterating if reached
                break

            # reset helper variables
            self.__sol["previous"] = self.__sol["current"].copy()
            if self.problem.has_exact():
                self.__err_vec["previous"] = self.__err_vec["current"].copy()
                self.__err_vec["current"] = np.array([0.0] * self.num_time_steps)
        # end while:_converged
        self.timer.stop()

        if _converged:
            LOG.info("# Converged after {:d} iteration(s).".format(_iter + 1))
            LOG.info("# Rel. Reduction: {:.3e}"
                     .format(_relred))
            if self.problem.has_exact():
                LOG.info("# Rel. Error: {:.3e}"
                         .format(_errred))
        else:
            warnings.warn("Explicit SDC: Did not converged!")
            LOG.info("# FAILED: Relative reduction after {:d} iteration(s). Rel. Reduction: {:.3e}"
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
