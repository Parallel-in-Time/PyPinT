# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import is_interactive

from pypint.plugins.plotters.i_plotter import IPlotter
from pypint.solvers.i_iterative_time_solver import IIterativeTimeSolver
from pypint.solvers.states import ISolverState
from pypint.solvers.diagnosis.norms import supremum_norm
from pypint.utilities import assert_is_key, assert_is_instance, assert_condition
from pypint import LOG


class ReductionResidualPlotter(IPlotter):
    """Plotts residual and reduction of multiple solutions of an iterative time solver

    See Also
    --------
    :py:class:`.IPlotter` : overridden class
    """

    _colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    _styles = np.array(['o', 'v', '^', '<', '>', 's', 'p', '*', '+', 'x', 'D'])

    def __init__(self, *args, **kwargs):
        super(ReductionResidualPlotter, self).__init__(args, **kwargs)
        self._solvers = None
        self._states = None
        self._nodes = None
        self.__reduction_limits = [0.0, 0.0]
        self.__residual_limits = [0.0, 0.0]

    def plot(self, *args, **kwargs):
        """Plots the solution and optional also the error for each iteration.

        If ``file_name`` has been specified on initialization (see :py:meth:`.IPlotter.__init__`) the plot is stored
        there.
        Otherwise an interactive plotting session is started.

        Parameters
        ----------
        solvers : :py:class:`.IIterativeTimeSolver`
            solver instances used to calculate the solutions

        states : :py:class:`.ISolverState`
            states of the solvers

        Raises
        ------
        ValueError

            * if ``solvers`` is not given or is not a :py:class:`numpy.ndarray` of :py:class:`.IIterativeTimeSolver`
            * if ``states`` is not given or is not a :py:class:`numpy.ndarray` of :py:class:`.ISolverState`
            * if ``states`` has more than 7 states
            * if the size of ``states`` does not equal the size of ``solvers``
        """
        super(ReductionResidualPlotter, self).plot(args, **kwargs)

        assert_is_key(kwargs, 'solvers', "Solver must be given", self)
        assert_is_instance(kwargs['solvers'], np.ndarray,
                           "Solver must be a numpy.ndarray: NOT %s" % kwargs['solvers'].__class__.__name__,
                           self)
        [assert_is_instance(_solver, IIterativeTimeSolver,
                            "All solvers must be an Iterative Time Solver: NOT %s" % _solver.__class__.__name__,
                            self) for _solver in kwargs['solvers']]
        self._solvers = kwargs['solvers']

        assert_is_key(kwargs, 'states', "States must be given", self)
        assert_is_instance(kwargs['states'], np.ndarray,
                           "States must be a list: NOT %s" % kwargs['states'].__class__.__name__,
                           self)
        assert_condition(kwargs['states'].size <= 7,
                         ValueError, "Can only handle up to 7 solutions: %d" % kwargs['states'].size,
                         self)
        [assert_is_instance(_state, ISolverState,
                            "All states must be an ISolverState: NOT %s" % _state.__class__.__name__,
                            self) for _state in kwargs['states']]
        self._states = kwargs['states']

        assert_condition(self._solvers.size == self._states.size,
                         ValueError, "Number of solvers must equal number of states: %d != %d"
                                     % (self._solvers.size, self._states.size),
                         self)

        self._nodes = self._states[0].first.time_points

        if self._solvers[0].problem.time_start != self._nodes[0]:
            self._nodes = np.concatenate(([self._solvers[0].problem.time_start], self._nodes))
        if self._solvers[0].problem.time_end != self._nodes[-1]:
            self._nodes = np.concatenate((self._nodes, [self._solvers[0].problem.time_end]))

        plt.title("Residuals and Reduction per Iteration for different Lambdas")
        self._plot_residuals_reductions()

        if self._file_name is not None:
            fig = plt.gcf()
            fig.set_dpi(300)
            fig.set_size_inches((15., 15.))
            LOG.debug("Plotting figure with size (w,h) {:s} inches and {:d} DPI."
                      .format(fig.get_size_inches(), fig.get_dpi()))
            fig.savefig(self._file_name)

        if is_interactive():
            plt.show(block=True)
        else:
            plt.close('all')

    def _plot_residuals_reductions(self):
        plt.hold(True)
        for _state in range(0, self._states.size):
            self._add_solution_plot(_state)
        LOG.debug("Plotting within {:s} x {:s}".format(self.__residual_limits, self.__reduction_limits))
        _limits = [0.0, 0.0]
        _limits[0] = min(self.__residual_limits[0], self.__reduction_limits[0])
        _limits[1] = max(self.__residual_limits[0], self.__reduction_limits[1])
        plt.xlabel("residual")
        plt.xlim(_limits)
        plt.ylabel("reduction")
        plt.ylim(_limits)
        #plt.legend(loc=4)
        plt.grid(True)

    def _add_solution_plot(self, index):
        _res = np.zeros(len(self._states[index]) - 1)
        _red = np.zeros(len(self._states[index]) - 1)
        for i in range(1, len(self._states[index])):
            _res[i - 1] = supremum_norm(self._states[index][i].solution.residuals[-1].value)
            _red[i - 1] = supremum_norm(self._states[index].solution.solution_reduction(i))

        if _res.min() < self.__residual_limits[0]:
            self.__residual_limits[0] = _res.min()
        if _res.max() > self.__residual_limits[1]:
            self.__residual_limits[1] = _res.max()

        if _red.min() < self.__reduction_limits[0]:
            self.__reduction_limits[0] = _red.min()
        if _red.max() > self.__reduction_limits[1]:
            self.__reduction_limits[1] = _red.max()

        plt.loglog(_res, _red, color=ReductionResidualPlotter._colors[index])
