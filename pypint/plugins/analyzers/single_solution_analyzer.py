# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_analyzer import IAnalyzer
from pypint.plugins.plotters.single_solution_plotter import SingleSolutionPlotter


class SingleSolutionAnalyzer(IAnalyzer):
    def __init__(self, *args, **kwargs):
        super(SingleSolutionAnalyzer, self).__init__(args, kwargs)
        self._solver = None
        self._plotter = SingleSolutionPlotter()

    def run(self):
        # plot the last solution
        self._plotter.plot(solver=self._solver,
                           solution=self._data,
                           errorplot=True)

    def add_data(self, *args, **kwargs):
        super(SingleSolutionAnalyzer, self).add_data(args, kwargs)
        if "solver" in kwargs:
            self._solver = kwargs["solver"]
        if "solution" in kwargs:
            self._data = kwargs["solution"]
