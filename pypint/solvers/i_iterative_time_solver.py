# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.solutions.iterative_solution import IterativeSolution
from pypint import LOG


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
        self._max_iterations = None
        self._threshold = None

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

            ``max_iterations`` : integer
                see :py:attr:`.max_iterations`
            ``threshold`` : float
                see :py:attr:`.threshold`
        """
        self._problem = problem
        self._integrator = integrator
        if "max_iterations" in kwargs:
            self.max_iterations = kwargs["max_iterations"]
        if "threshold" in kwargs:
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
    def max_iterations(self):
        """
        Summary
        -------
        Accessor for the maximum number of iterations for this solver.

        Extended Summary
        ----------------
        The solver will never carry out more iterations than this number.

        Parameters
        ----------
        max_iterations : integer
            Maximum iterations of this solver.

        Returns
        -------
        maximum iterations : integer
            Maximum iterations of this solver.
        """
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, max_iterations):
        self._max_iterations = max_iterations

    @property
    def threshold(self):
        """
        Summary
        -------
        Accessor for general threshold of this solver.

        Extended Summary
        ----------------
        Depending on the solver's algorithm the threshold is used in multiple ways to create
        termination conditions.

        Parameters
        ----------
        threshold : float
            Desired threshold.

        Returns
        -------
        threshold : float
            Desired threshold.
        """
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold

    @property
    def integrator(self):
        return self._integrator
