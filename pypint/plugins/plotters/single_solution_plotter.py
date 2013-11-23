# coding=utf-8

"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_plotter import IPlotter
import numpy as np
import matplotlib.pyplot as plt
from pypint.utilities import func_name


class SingleSolutionPlotter(IPlotter):
    def __init__(self, *args, **kwargs):
        super(SingleSolutionPlotter, self).__init__(args, kwargs)

    def plot(self, *args, **kwargs):
        super(SingleSolutionPlotter, self).plot(args, kwargs)
        if "solver" not in kwargs or "solution" not in kwargs:
            raise ValueError(func_name(self) +
                             "Both, solver and solution, must be given.")
        solver = kwargs["solver"]
        solution = kwargs["solution"]
        nodes = solver.integrator.nodes
        if solver.problem.time_start != nodes[0]:
            nodes = np.concatenate(([solver.problem.time_start], nodes))
        if solver.problem.time_end != nodes[-1]:
            nodes = np.concatenate((nodes, [solver.problem.time_end]))

        plt.suptitle(r"after {:d} iterations; overall reduction: {:.2e}"
                     .format(solution.used_iterations, solution.reduction))
        plt.subplot(2, 1, 1)
        if "title" in kwargs:
            plt.title(kwargs["title"])
        if solver.problem.has_exact() and solution.errors[-1].max() > 1e-2:
            self._final_solution(solution, nodes, exact=[[solver.problem.exact(0.0, node)] for node in nodes])
        else:
            self._final_solution(solution, nodes)
        plt.subplot(2, 1, 2)
        self._error_plot(solution, nodes)
        plt.show()

    def _final_solution(self, solution, nodes, *args, **kwargs):
        if "exact" in kwargs:
            plt.plot(nodes, solution.solution(), nodes, kwargs["exact"])
        else:
            plt.plot(nodes, solution.solution())
        plt.xticks(nodes)
        plt.xlabel("integration nodes")
        plt.ylabel(r'$u(t, \phi_t)$')
        plt.xlim(nodes[0], nodes[-1])
        plt.grid(True)

    def _error_plot(self, solution, nodes):
        errors = solution.errors
        for i in range(0, errors.size):
            plt.plot(nodes[1:], errors[i], label=r"Iteraion {:d}".format(i+1))
        plt.xticks(nodes)
        plt.xlim(nodes[0], nodes[-1])
        plt.yscale("log")
        plt.xlabel("integration nodes")
        plt.ylabel(r'absolute error of iterations')
        #plt.legend(loc="upper center", fontsize="x-small")
        plt.grid(True)
