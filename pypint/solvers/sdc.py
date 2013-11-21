# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_iterative_time_solver import IIterativeTimeSolver
import numpy as np
from pypint.integrators.sdc_integrator import SdcIntegrator
from pypint.integrators.node_providers.gauss_lobatto_nodes \
    import GaussLobattoNodes
from pypint.integrators.weight_function_providers.polynomial_weight_function \
    import PolynomialWeightFunction
from pypint.problems.i_initial_value_problem import IInitialValueProblem
from pypint.solutions.iterative_solution import IterativeSolution
from pypint.utilities import func_name
from pypint import LOG


class Sdc(IIterativeTimeSolver):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.max_iterations = 5
        self.min_reduction = 1e-7
        self._num_time_steps = 2
        self.__prev_sol = np.zeros(0)
        self.__curr_sol = np.zeros(0)
        self.__delta_times = np.zeros(0)
        self.__delta_interval = 0
        self.__smat = np.zeros(0)

    def init(self, problem, integrator=SdcIntegrator(), **kwargs):
        """
        Parameters
        ----------
        kwargs : further named arguments
            In addition to the options available via :py:meth:`.IIterativeTimeSolver.init()`
            the following specific options are available:

            ``num_steps`` : integer
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
        if "num_steps" not in kwargs:
            kwargs["num_steps"] = self.num_time_steps
        if "nodes_type" not in kwargs:
            kwargs["nodes_type"] = GaussLobattoNodes()
        if "weights_type" not in kwargs:
            kwargs["weights_type"] = PolynomialWeightFunction()

        # initialize integrator
        self._integrator.init(kwargs["nodes_type"], kwargs["num_steps"] + 1, kwargs["weights_type"],
                              np.array([self.problem.time_start, self.problem.time_end]))

        # initialize helper variables
        self.__prev_sol = np.array([self.problem.initial_value] * (self.num_time_steps + 1))
        self.__curr_sol = np.zeros(self.num_time_steps + 1)

        # compute time step distances
        self.__delta_interval = self.problem.time_end - self.problem.time_start
        self.__delta_times = np.zeros(self.num_time_steps)
        for i in range(0, self.num_time_steps):
            self.__delta_times[i] = self._integrator.nodes[i+1] - self._integrator.nodes[i]

    def run(self):
        _sol = IterativeSolution()
        for k in range(0, self.max_iterations):
            for step in range(0, self.num_time_steps):
                _dt = self.__delta_times[step]
                _time = self._integrator.nodes[step]

                # gather values for integration
                _copy_mask = np.concatenate((np.array([True] * step),
                                             np.array([False] * (self.num_time_steps - step + 1))))
                _integrate_values = np.where(_copy_mask, self.__curr_sol, self.__prev_sol)

                # evaluate problem for integration values
                _integrate_values = np.array([self.problem.evaluate(self._integrator.nodes[step], val)
                                              for val in _integrate_values])

                # integrate
                integral = self._integrator.evaluate(_integrate_values, until_node_index=step)

                # compute step
                self.__curr_sol[step + 1] = \
                    self.__curr_sol[step] + \
                    _dt * (self.problem.evaluate(_time, self.__curr_sol[step]) -
                           self.problem.evaluate(_time, self.__prev_sol[step])) + \
                    self.__delta_interval * integral
            # end for:step
            _sol.add_solution(k, self.__curr_sol)
            self.__prev_sol = self.__curr_sol.copy()
            self.__curr_sol = np.zeros(self.num_time_steps + 1)
        # end for:iter
        return _sol

    @property
    def num_time_steps(self):
        return self._num_time_steps
