# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.solvers.states.i_solver_state import ISolverState
from pypint.solvers.cores.i_solver_core import ISolverCore
from pypint.utilities.threshold_check import ThresholdCheck
from pypint.utilities import assert_condition, assert_is_callable, class_name


class IIterativeTimeSolver(object):
    """Basic interface for iterative time solvers.
    """

    def __init__(self, *args, **kwargs):
        self._problem = None
        self._integrator = None
        self._core = ISolverCore()
        self._timer = None
        self._threshold_check = ThresholdCheck()
        self._state = ISolverState()

    def init(self, problem, **kwargs):
        """Initializes the solver with a given problem and options.

        Parameters
        ----------
        problem : :py:class:`.IProblem`
            The problem this solver should solve.

        integrator : :py:class:`.IntegratorBase`
            Integrator to be used by this solver.

        threshold : :py:class:`.ThresholdCheck`
            *(optional)*
            see :py:attr:`.threshold`
        """
        self._problem = problem
        if 'integrator' in kwargs:
            assert_is_callable(kwargs['integrator'], message="Integrator must be instantiable.", checking_obj=self)
            self._integrator = kwargs['integrator']()
        if "threshold" in kwargs and isinstance(kwargs["threshold"], ThresholdCheck):
            self.threshold = kwargs["threshold"]

    def run(self, core, **kwargs):
        """Applies this solver.

        Parameters
        ----------
        solution_type : :py:class:`tuple` of two :py:class:`class`
            Tuple of two classes specifying the solution type and underlying solution storage data type.
            The first item must be a class derived off :py:class:`.ISolution` and the second a class derived
            off :py:class:`.ISolutionData`.

        Returns
        -------
        solution : :py:class:`.ISolution`
            The solution of the problem.
        """
        assert_condition(issubclass(core, ISolverCore),
                         ValueError, message="The given solver core class must be valid: NOT {:s}"
                                             .format(class_name(core)),
                         checking_obj=self)
        self._core = core()

    @property
    def problem(self):
        """Accessor for the stored problem.

        Returns
        -------
        stored problem : :py:class:`.IProblem` or :py:class:`None`
            Stored problem after call to :py:meth:`.init` or :py:class:`None` if no problem was initialized.
        """
        return self._problem

    @property
    def state(self):
        """Read-only accessor for the sovler's state

        Returns
        -------
        state : :py:class:`.ISolverState`
        """
        return self._state

    @property
    def timer(self):
        """Accessor for the timer

        Parameters
        ----------
        timer : :py:class:`.TimerBase`

        Returns
        -------
        timer : :py:class:`.TimerBase`
        """
        return self._timer

    @timer.setter
    def timer(self, timer):
        self._timer = timer

    @property
    def threshold(self):
        """Accessor for threshold check of this solver.

        Depending on the solver's algorithm the threshold is used in multiple ways to check for termination conditions.

        Parameters
        ----------
        threshold : :py:class:`.ThresholdCheck`
            Desired threshold.

        Returns
        -------
        threshold : :py:class:`.ThresholdCheck`
            Stored and used threshold.
        """
        return self._threshold_check

    @threshold.setter
    def threshold(self, threshold):
        self._threshold_check = threshold

    @property
    def integrator(self):
        """Read-only accessor for the used integrator

        Returns
        -------
        integrator : :py:class:`.IntegratorBase`
        """
        return self._integrator

    def print_lines_for_log(self):
        _lines = {
            'Integrator': self.integrator.print_lines_for_log(),
            'Thresholds': self.threshold.print_lines_for_log()
        }
        return _lines

    def _print_header(self):
        pass

    def _print_footer(self):
        pass


__all__ = ['IIterativeTimeSolver']
