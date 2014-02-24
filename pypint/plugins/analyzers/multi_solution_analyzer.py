# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np

from pypint.plugins.analyzers.i_analyzer import IAnalyzer
from pypint.plugins.plotters.reduction_residual_plotter import ReductionResidualPlotter
from pypint.solvers.i_iterative_time_solver import IIterativeTimeSolver
from pypint.solvers.states import ISolverState
from pypint.utilities import assert_is_key, assert_is_instance


class MultiSolutionAnalyzer(IAnalyzer):
    """Analyzer for multiple solver states

    For now, it only plots the final residual vs. final reduction of all given states.
    Only up to seven separate states are supported.
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        plotter_options : :py:class:`dict`
            options to be passed on to the plotter
        """
        super(MultiSolutionAnalyzer, self).__init__(args, **kwargs)

        if 'plotter_options' in kwargs:
            self._plotter = ReductionResidualPlotter(**kwargs['plotter_options'])
        else:
            self._plotter = ReductionResidualPlotter()

        self._solvers = []
        self._data = []

    def run(self, **kwargs):
        """Executes the analysis
        """
        super(MultiSolutionAnalyzer, self).run(**kwargs)

        # plot the last solution
        self._plotter.plot(solvers=np.array(self._solvers),
                           states=np.array(self._data))

    def add_data(self, *args, **kwargs):
        """
        Parameters
        ----------
        solver : :py:class:`.IIterativeTimeSolver`
            solver

        state : :py:class:`.ISolverState`
            state of the solver

        Raises
        ------
        ValueError

            * if ``solver`` is not given or is not a :py:class:`.IIterativeTimeSolver`
            * if ``state`` is not given or is not a :py:class:`.ISolverState`
        """
        super(MultiSolutionAnalyzer, self).add_data(args, kwargs)

        assert_is_key(kwargs, 'solver', "Solver must be given", self)
        assert_is_instance(kwargs['solver'], IIterativeTimeSolver,
                           "Solver must be an Iterative Time Solver: NOT %s" % kwargs['solver'].__class__.__name__,
                           self)

        assert_is_key(kwargs, 'state', "State must be given", self)
        assert_is_instance(kwargs['state'], ISolverState,
                           "Given state must be a ISolverState: NOT %s" % kwargs['state'].__class__.__name__,
                           self)

        self._solvers.append(kwargs['solver'])
        self._data.append(kwargs['state'])
