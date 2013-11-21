# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class IIterativeTimeSolver(object):
    """
    Summary
    -------
    Basic interface for iterative time solvers.
    """

    def __init__(self):
        self._problem = None
        self._integrator = None
        self._timer = None
        self._max_iterations = -1
        self._min_reduction = -1

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
            ``min_reduction`` : integer
                see :py:attr:`.min_reduction`
        """
        self._problem = problem
        self._integrator = integrator
        if "max_iterations" in kwargs:
            self.max_iterations = kwargs["max_iterations"]
        if "min_reduction" in kwargs:
            self.min_reduction = kwargs["min_reduction"]

    def run(self):
        """
        Summary
        -------
        Applies this solver.

        Extended Summary
        ----------------
        It is guaranteed that the solver does not carry out more than
        :py:attr:`.max_iterations` iterations.
        In case the desired :py:attr:`.min_reduction` is reached, the solver
        will abort prior reaching :py:attr:`.max_iterations`.

        Returns
        -------
        solution : :py:class:`.ISolution`
            The solution of the problem.
        """
        return None

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
    def min_reduction(self):
        """
        Summary
        -------
        Accessor for the minimum reduction of this solver.

        Extended Summary
        ----------------
        The solver will try to reach the specified minimum error reduction by
        additional iterations, not exceeding :py:attr:`.max_iterations`.

        Parameters
        ----------
        min_reduction : float
            Desired minimum error reduction.

        Returns
        -------
        minimum reduction : float
            Desired minimum error reduction.
        """
        return self._min_reduction

    @min_reduction.setter
    def min_reduction(self, min_reduction):
        self._min_reduction = min_reduction
