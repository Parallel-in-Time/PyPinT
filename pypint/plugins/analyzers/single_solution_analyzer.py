# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.plugins.analyzers.i_analyzer import IAnalyzer
from pypint.plugins.plotters.single_solution_plotter import SingleSolutionPlotter
from pypint.solvers.i_iterative_time_solver import IIterativeTimeSolver
from pypint.solvers.states import ISolverState
from pypint.utilities import assert_named_argument


class SingleSolutionAnalyzer(IAnalyzer):
    """Analyzer for a single solution instance.

    For now, it only plots the final solution and the error of each iteration.
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        plotter_options : :py:class:`dict`
            options to be passed on to the plotter
        """
        super(SingleSolutionAnalyzer, self).__init__(args, **kwargs)
        self._solver = None
        if 'plotter_options' in kwargs:
            self._plotter = SingleSolutionPlotter(**kwargs['plotter_options'])
        else:
            self._plotter = SingleSolutionPlotter()

    def run(self, **kwargs):
        """
        Parameters
        ----------
        solver : :py:class:`.IIterativeTimeSolver`

        Raises
        ------
        ValueError
            if ``solver`` is not given and not an :py:class:`.IIterativeTimeSolver`
        """
        super(SingleSolutionAnalyzer, self).run(**kwargs)

        assert_named_argument('solver', kwargs, types=IIterativeTimeSolver, descriptor="Solver", checking_obj=self)

        # plot the last solution
        self._plotter.plot(solver=kwargs['solver'],
                           state=self._data,
                           errorplot=True,
                           residualplot=True)

    def add_data(self, *args, **kwargs):
        """
        Parameters
        ----------
        state : :py:class:`.ISolverState`
            state of the solver

        Raises
        ------
        ValueError
            if ``state`` is not given or is not a :py:class:`.ISolverState`
        """
        super(SingleSolutionAnalyzer, self).add_data(args, kwargs)
        assert_named_argument('state', kwargs, types=ISolverState, descriptor="State", checking_obj=self)
        self._data = kwargs['state']
