# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import is_interactive

from pypint.plugins.plotters.i_plotter import IPlotter
from pypint.plugins.plotters import colorline
from pypint.problems import problem_has_exact_solution
from pypint.solvers.i_iterative_time_solver import IIterativeTimeSolver
from pypint.solvers.states import ISolverState
from pypint.solvers.diagnosis.norms import supremum_norm
from pypint.utilities import assert_is_key, assert_is_instance
from pypint import LOG


class SingleSolutionPlotter(IPlotter):
    """Plotter for a single solution of an iterative time solver.

    See Also
    --------
    :py:class:`.IPlotter` : overridden class
    """
    def __init__(self, *args, **kwargs):
        super(SingleSolutionPlotter, self).__init__(args, **kwargs)
        self._solver = None
        self._state = None
        self._nodes = None
        self._errplot = False
        self._residualplot = False

    def plot(self, *args, **kwargs):
        """Plots the solution and optional also the error for each iteration.

        Parameters
        ----------
        solver : :py:class:`.IIterativeTimeSolver`
            solver instance used to calculate the solution

        state : :py:class:`.ISolverState`
            state containing information to plot

        errplot : :py:class:`bool`
            *(optional)*
            if given and :py:class:`True` also plots the errors for each iteration found in the solution

        residualplot : :py:class:`bool`
            *(optional)*
            if given and :py:class:`True` also plots the residual for each iteration found in the solution

        Raises
        ------
        ValueError

            * if ``solver`` not given and not an :py:class:`.IIterativeTimeSolver`
            * if ``state`` not given and not an :py:class:`.ISolverState`
        """
        super(SingleSolutionPlotter, self).plot(args, **kwargs)

        assert_is_key(kwargs, 'solver', "Solver must be given", self)
        assert_is_instance(kwargs['solver'], IIterativeTimeSolver,
                           "Solver must be an Iterative Time Solver: NOT %s" % kwargs['solver'].__class__.__name__,
                           self)
        assert_is_key(kwargs, 'state', "State must be given", self)
        assert_is_instance(kwargs['state'], ISolverState,
                           "State must be an ISolverState: NOT %s" % kwargs['state'].__class__.__name__,
                           self)

        self._solver = kwargs['solver']
        self._state = kwargs['state']
        self._nodes = self._state.first.time_points

        _subplots = 1
        _curr_subplot = 0
        if 'errorplot' in kwargs and kwargs['errorplot']:
            _subplots += 1
            self._errplot = True
        if 'residualplot' in kwargs and kwargs['residualplot']:
            _subplots += 1
            self._residualplot = True

        if self._solver.problem.time_start != self._nodes[0]:
            self._nodes = np.concatenate(([self._solver.problem.time_start], self._nodes))
        if self._solver.problem.time_end != self._nodes[-1]:
            self._nodes = np.concatenate((self._nodes, [self._solver.problem.time_end]))

        if self._errplot or self._residualplot:
            plt.suptitle(r"after {:d} iterations; overall reduction: {:.2e}"
                         .format(len(self._state),
                                 supremum_norm(self._state.solution
                                               .solution_reduction(self._state.last_iteration_index))))
            _curr_subplot += 1
            plt.subplot(_subplots, 1, _curr_subplot)

        self._final_solution()
        plt.title(self._solver.problem.__str__())

        if self._errplot:
            _curr_subplot += 1
            plt.subplot(3, 1, _curr_subplot)
            self._error_plot()

        if self._residualplot:
            _curr_subplot += 1
            plt.subplot(3, 1, _curr_subplot)
            self._residual_plot()

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

    def _final_solution(self):
        _solution = np.insert(self._state.last_iteration.solution.values, 0, [self._state.initial.solution.value],
                              axis=0)

        if self._state.last_iteration.solution.numeric_type == np.complex:
            colorline(np.array([_p.real for _p in _solution[:, 0]], dtype=np.float),
                      np.array([_p.imag for _p in _solution[:, 0]], dtype=np.float))
        else:
            plt.plot(self._nodes, _solution[:, 0], label="Solution")
        # TODO fix error checking
        # if problem_has_exact_solution(self._solver.problem, self):
                # and self._state.last_iteration.solution.errors.max() > 1e-2:
            # exact = np.array([self._solver.problem.exact(_t) for _t in self._nodes],
            #                  dtype=self._solver.problem.numeric_type)
            # if exact.dtype == np.complex:
            #     LOG.debug("    plotting exact solution as complex values")
            #     colorline(exact.real, exact.imag)
            # else:
            #     LOG.debug("    plotting exact solution as real values")
            #     plt.plot(self._nodes, exact, label="Exact")
        if self._state.last_iteration.solution.numeric_type == np.complex:
            plt.xlabel("real")
            plt.ylabel("imag")
        else:
            plt.xticks(self._nodes)
            plt.xlabel("integration nodes")
            plt.ylabel(r'$u(t, \phi_t)$')
            plt.xlim(self._nodes[0], self._nodes[-1])
            plt.legend()
        plt.grid(True)

    def _error_plot(self):
        for i in range(0, len(self._state)):
            _error = np.insert(np.array([_e.value for _e in self._state[i].solution.errors]), 0, [0.0], axis=0)
            plt.plot(self._nodes, _error, label=r"Iteraion {:d}".format(i+1))
        plt.xticks(self._nodes)
        plt.xlim(self._nodes[0], self._nodes[-1])
        plt.yscale("log")
        plt.xlabel("integration nodes")
        plt.ylabel(r'absolute error of iterations')
        plt.grid(True)

    def _residual_plot(self):
        for i in range(0, len(self._state)):
            _residual = np.insert(np.array([_r.value for _r in self._state[i].solution.residuals]), 0, [0.0], axis=0)
            plt.plot(self._nodes, _residual, label=r"Iteration {:d}".format(i+1))
        plt.xticks(self._nodes)
        plt.xlim(self._nodes[0], self._nodes[-1])
        plt.yscale("log")
        plt.xlabel("integration nodes")
        plt.ylabel(r'residual')
        plt.grid(True)
