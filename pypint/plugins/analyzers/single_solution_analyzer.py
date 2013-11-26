# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_analyzer import IAnalyzer
from pypint.plugins.plotters.single_solution_plotter import SingleSolutionPlotter


class SingleSolutionAnalyzer(IAnalyzer):
    """
    Summary
    -------
    Analyzer for a single solution instance.

    Extended Summary
    ----------------
    For now, it only plots the final solution and the error of each iteration.
    """
    def __init__(self, *args, **kwargs):
        super(SingleSolutionAnalyzer, self).__init__(args, **kwargs)
        self._solver = None
        if "plotter_file_name" in kwargs:
            self._plotter = SingleSolutionPlotter(file_name=kwargs["plotter_file_name"])
        else:
            self._plotter = SingleSolutionPlotter()

    def run(self):
        # plot the last solution
        self._plotter.plot(solver=self._solver,
                           solution=self._data,
                           errorplot=True,
                           residualplot=True)

    def add_data(self, *args, **kwargs):
        """
        Parameters
        ----------
        solver : IIterativeTimeSolver
            Solver instance used to calculate the solution to analyze.

        solution : ISolution
            Solution returned by the solver.
        """
        super(SingleSolutionAnalyzer, self).add_data(args, kwargs)
        if "solver" in kwargs:
            self._solver = kwargs["solver"]
        if "solution" in kwargs:
            self._data = kwargs["solution"]
