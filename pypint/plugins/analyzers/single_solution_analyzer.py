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
        if "title" in kwargs:
            self._title = kwargs["title"]
        else:
            self._title = "Solution"

    def run(self):
        # plot the last solution
        self._plotter.plot(solver=self._solver,
                           solution=self._data,
                           title=self._title)

    def add_data(self, *args, **kwargs):
        super(SingleSolutionAnalyzer, self).add_data(args, kwargs)
        if "solver" in kwargs:
            self._solver = kwargs["solver"]
        if "solution" in kwargs:
            self._data = kwargs["solution"]
