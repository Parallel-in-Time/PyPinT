# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.plugins.analyzers.i_analyzer import IAnalyzer
from pypint.plugins.plotters.single_solution_plotter import SingleSolutionPlotter


class SingleSolutionAnalyzer(IAnalyzer):
    """Analyzer for a single solution instance.

    For now, it only plots the final solution and the error of each iteration.
    """
    def __init__(self, *args, **kwargs):
        super(SingleSolutionAnalyzer, self).__init__(args, **kwargs)
        self._solver = None
        if "plotter_options" in kwargs:
            self._plotter = SingleSolutionPlotter(**kwargs["plotter_options"])
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
        solver : :py:class:`.IIterativeTimeSolver`
            Solver instance used to calculate the solution to analyze.

        solution : :py:class:`.ISolution`
            Solution returned by the solver.
        """
        super(SingleSolutionAnalyzer, self).add_data(args, kwargs)
        if "solver" in kwargs:
            self._solver = kwargs["solver"]
        if "solution" in kwargs:
            self._data = kwargs["solution"]
