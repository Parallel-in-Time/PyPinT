# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from ..solutions.iterative_solution import IterativeSolution
from ..utilities.threshold_check import ThresholdCheck
from ..utilities import assert_condition


class IIterativeTimeSolver(object):
    """
    Summary
    -------
    Basic interface for iterative time solvers.
    """

    class State(object):
        """
        Summary
        -------
        Internal class holding solver iteration states such as intermediate results.
        """

        num_points = 0

        def __init__(self, iteration=0):
            self._iteration = iteration
            self._solution = None
            self._error = None
            self._residual = None
            self._reduction_of_solution = None
            self._reduction_of_error = None

        @property
        def iteration(self):
            return self._iteration
        @iteration.setter
        def iteration(self, iteration):
            assert_condition(iteration > 0,
                             ValueError, "Iteration count must be possitive: {:d}".format(iteration),
                             self)
            self._iteration = iteration

        @property
        def solution(self):
            return self._solution
        @solution.setter
        def solution(self, solution):
            self._solution = solution.copy()

        @property
        def error(self):
            return self._error
        @error.setter
        def error(self, error):
            self._error = error

        @property
        def residual(self):
            return self._residual
        @residual.setter
        def residual(self, residual):
            self._residual = residual

        @property
        def reduction_of_solution(self):
            return self._reduction_of_solution
        @reduction_of_solution.setter
        def reduction_of_solution(self, reduction_of_solution):
            self._reduction_of_solution = reduction_of_solution

        @property
        def reduction_of_error(self):
            return self._reduction_of_error
        @reduction_of_error.setter
        def reduction_of_error(self, reduction_of_error):
            self._reduction_of_error = reduction_of_error

    def __init__(self, *args, **kwargs):
        self._problem = None
        self._integrator = None
        self._timer = None
        self._threshold_check = ThresholdCheck()
        self._states = []

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
    def states(self):
        return self._states

    @property
    def initial_state(self):
        return self._states[0]

    @property
    def current_state(self):
        return self._states[-1]

    @property
    def previous_state(self):
        return self._states[-2]


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
