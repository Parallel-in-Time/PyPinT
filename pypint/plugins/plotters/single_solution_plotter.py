# coding=utf-8

"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_plotter import IPlotter
import matplotlib.pyplot as plt
from pypint.utilities import func_name


class SingleSolutionPlotter(IPlotter):
    def __init__(self, *args, **kwargs):
        super(SingleSolutionPlotter, self).__init__(args, kwargs)

    def plot(self, *args, **kwargs):
        super(SingleSolutionPlotter, self).plot(args, kwargs)
        if "integrator" not in kwargs or "solution" not in kwargs:
            raise ValueError(func_name(self) +
                             "Both, integrator and solution, must be given.")
        integrator = kwargs["integrator"]
        solution = kwargs["solution"]
        plt.plot(integrator.nodes, solution.solution())
        plt.xticks(integrator.nodes)
        if "title" in kwargs:
            plt.title(kwargs["title"])
        plt.suptitle(r"after {:d} iterations; overall reduction: {:.2e}"
                     .format(solution.used_iterations, solution.reduction))
        plt.xlabel("integration nodes")
        plt.ylabel(r'$u(t, \phi_t)$')
        plt.grid(True)
        plt.show()
