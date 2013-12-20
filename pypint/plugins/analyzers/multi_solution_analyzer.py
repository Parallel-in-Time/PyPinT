# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_analyzer import IAnalyzer
import numpy as np
from pypint.plugins.plotters.reduction_residual_plotter import ReductionResidualPlotter


class MultiSolutionAnalyzer(IAnalyzer):
    """
    Summary
    -------
    Analyzer for multiple solution instance.

    Extended Summary
    ----------------
    For now, it only plots the final solution and the error of each iteration.
    """
    def __init__(self, *args, **kwargs):
        super(MultiSolutionAnalyzer, self).__init__(args, **kwargs)
        self._solver = None
        if "plotter_options" in kwargs:
            self._plotter = ReductionResidualPlotter(**kwargs["plotter_options"])
        else:
            self._plotter = ReductionResidualPlotter()
        self._data = []

    def run(self):
        # plot the last solution
        self._plotter.plot(solver=self._solver,
                           solutions=np.array(self._data))

    def add_data(self, *args, **kwargs):
        """
        Parameters
        ----------
        solver : IIterativeTimeSolver
            Solver instance used to calculate the solution to analyze.

        solution : ISolution
            Solution returned by the solver.
        """
        super(MultiSolutionAnalyzer, self).add_data(args, kwargs)
        if "solver" in kwargs:
            self._solver = kwargs["solver"]
        if "solution" in kwargs:
            self._data.append(kwargs["solution"])