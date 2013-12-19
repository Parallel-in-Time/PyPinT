# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_plotter import IPlotter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import is_interactive
from pypint.utilities import critical_assert
from pypint import LOG


class ReductionResidualPlotter(IPlotter):
    """
    Summary
    -------
    Plotts residual and reduction of multiple solutions of an iterative time solver.

    See Also
    --------
    .IPlotter
        overridden class
    """

    _colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    _styles = np.array(['o', 'v', '^', '<', '>', 's', 'p', '*', '+', 'x', 'D'])

    def __init__(self, *args, **kwargs):
        super(ReductionResidualPlotter, self).__init__(args, **kwargs)
        self._solver = None
        self._solutions = None
        self._nodes = None
        self.__reduction_limits = [0.0, 0.0]
        self.__residual_limits = [0.0, 0.0]

    def plot(self, *args, **kwargs):
        """
        Summary
        -------
        Plots the solution and optional also the error for each iteration.

        Parameters
        ----------
        solver : IIterativeTimeSolver
            The solver instance used to calculate the solution.

        solution : ISolution
            The solution.
        """
        super(ReductionResidualPlotter, self).plot(args, **kwargs)

        critical_assert("solver" in kwargs and "solutions" in kwargs,
                        ValueError, "Both, solver and solutions, must be given.", self)

        self._solver = kwargs["solver"]
        critical_assert(isinstance(kwargs["solutions"], np.ndarray),
                        ValueError, "Solutions must be a numpy.ndarray of solutions.", self)
        critical_assert(kwargs["solutions"].size <= 7,
                        ValueError, "Can only handle up to 7 solutions: {:d}".format(kwargs["solutions"].size), self)
        self._solutions = kwargs["solutions"]
        self._nodes = self._solutions[0].points

        if self._solver.problem.time_start != self._nodes[0]:
            self._nodes = np.concatenate(([self._solver.problem.time_start], self._nodes))
        if self._solver.problem.time_end != self._nodes[-1]:
            self._nodes = np.concatenate((self._nodes, [self._solver.problem.time_end]))

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
        for sol in range(0, self._solutions.size):
            self._add_solution_plot(sol)
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
        _residuals = self._solutions[index].residuals
        _reductions = self._solutions[index].reductions
        _res = np.zeros(len(_residuals) - 1)
        _red = np.zeros(len(_residuals) - 1)
        for i in range(1, len(_residuals)):
            _res[i - 1] = _residuals[i][-1]
            _red[i - 1] = _reductions["solution"][i]

        if _res.min() < self.__residual_limits[0]:
            self.__residual_limits[0] = _res.min()
        if _res.max() > self.__residual_limits[1]:
            self.__residual_limits[1] = _res.max()

        if _red.min() < self.__reduction_limits[0]:
            self.__reduction_limits[0] = _red.min()
        if _red.max() > self.__reduction_limits[1]:
            self.__reduction_limits[1] = _red.max()

        plt.loglog(_res, _red, color=ReductionResidualPlotter._colors[index])
