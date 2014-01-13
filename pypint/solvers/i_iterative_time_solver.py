# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.solutions.iterative_solution import IterativeSolution
from pypint.utilities.threshold_check import ThresholdCheck


class IIterativeTimeSolver(object):
    """
    Summary
    -------
    Basic interface for iterative time solvers.
    """

    def __init__(self, *args, **kwargs):
        self._problem = None
        self._integrator = None
        self._timer = None
        self._threshold_check = ThresholdCheck()

    def init(self, problem, integrator, **kwargs):
        """
        Summary
        -------
        Initializes the solver with a given problem and options.

        Parameters
        ----------
        problem : :py:class:`.IProblem`
            The problem this solver should solve.

        integrator : :py:class:`.IntegratorBase`
            Integrator to be used by this solver.

        kwargs : further named arguments
            Supported names:

            ``threshold`` : :py:class:`.ThresholdCheck`
                see :py:attr:`.threshold`
        """
        self._problem = problem
        self._integrator = integrator
        if "threshold" in kwargs and isinstance(kwargs["threshold"], ThresholdCheck):
            self.threshold = kwargs["threshold"]

    def run(self, solution_class=IterativeSolution):
        """
        Summary
        -------
        Applies this solver.

        Returns
        -------
        solution : :py:class:`.ISolution`
            The solution of the problem.
        """
        return IterativeSolution()

    @property
    def problem(self):
        """
        Summary
        -------
        Accessor for the stored problem.

        Returns
        -------
        stored problem : :py:class:`.IProblem` or ``None``
            Stored problem after call to :py:meth:`.init` or ``None`` if no
            problem was initialized.
        """
        return self._problem

    @property
    def timer(self):
        return self._timer

    @timer.setter
    def timer(self, timer):
        self._timer = timer

    @property
    def threshold(self):
        """
        Summary
        -------
        Accessor for threshold check of this solver.

        Extended Summary
        ----------------
        Depending on the solver's algorithm the threshold is used in multiple ways to check for
        termination conditions.

        Parameters
        ----------
        threshold : :py:class:`.ThresholdCheck`
            Desired threshold.

        Returns
        -------
        threshold : :py:class:`.ThresholdCheck`
            Stored and used threshold.
        """
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold

    @property
    def integrator(self):
        return self._integrator

    def _print_header(self):
        pass

    def _print_footer(self):
        pass
