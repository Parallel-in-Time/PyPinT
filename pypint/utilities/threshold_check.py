# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from collections import OrderedDict

import numpy as np

from pypint.solvers.diagnosis import IDiagnosisValue
from pypint.solvers.diagnosis.norms import supremum_norm
from pypint.utilities import assert_condition, func_name
from pypint.utilities.logging import LOG


class ThresholdCheck(object):
    """Threshold Checking Handler
    """

    _default_min_threshold = 1e-7
    """Default minimum threshold
    """

    _default_max_threshold = 10
    """Default maximum threshold
    """

    _default_min_conditions = {
        'error reduction': _default_min_threshold,
        'solution reduction': _default_min_threshold,
        'residual': _default_min_threshold,
        'error': _default_min_threshold
    }

    _default_max_conditions = {
        'iterations': _default_max_threshold
    }

    def __init__(self, min_threshold=_default_min_threshold, max_threshold=_default_max_threshold,
                 conditions=('solution reduction', 'iterations')):
        """
        Parameters
        ----------
        min_threshold : :py:class:`float`
            threshold value for minimum criteria

        max_threshold : :py:class:`int`
            threshold value for maximum criteria

        conditions : :py:class:`tuple` of :py:class:`str`
            Tuple of strings defining the active criteria.
            Possible values are:

            * "``error reduction``"
            * "``solution reduction``"
            * "``residual``"
            * "``error``"
            * "``iterations``"

            (defaults to: ``('residual', 'iterations')``)
        """
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._conditions = {}
        self._set_conditions(conditions)
        self._reason = None

    def check(self, state):
        """Checks thresholds of given state

        Parameters
        ----------
        state : :py:class:`.ISolverState`
        """
        self._reason = []
        self._check_reduction(state)
        self._check_minimum('residual', state.current_iteration.final_step.solution.residual)
        self._check_minimum('error', state.current_iteration.final_step.solution.error)
        self._check_maximum('iterations', state.current_iteration_index + 1)
        if len(self._reason) == 0:
            self._reason = None

    def has_reached(self, log=False, human=False):
        """Gives list of thresholds reached

        Parameters
        ----------
        human : :py:class:`bool`
            if :py:class:`True` returns a human readable string listing reached thresholds
            (default :py:class:`False`)

        Returns
        -------
        reached_thresholds : :py:class:`list` or :py:class:`None`
            :py:class:`list` of matched thresholds or :py:class:`None` if non reached
        """
        if human:
            return "Threshold condition(s) met: {:s}".format(self._reason)
        if log:
            _r = OrderedDict()
            _r['Threshold condition(s) met'] = ', '.join(self._reason)
            return _r
        else:
            return self._reason

    @property
    def min_error_reduction(self):
        """Read-only accessor for the minimum reduction threshold for the error

        Returns
        -------
        reduction_threshold : :py:class:`float` or :py:class:`None`
            :py:class:`None` if reduction of error is not a criteria
        """
        if 'error reduction' in self._conditions:
            return self._conditions['error reduction']
        else:
            return None

    @property
    def min_solution_reduction(self):
        """Read-only accessor for the minimum reduction threshold for the solution

        Returns
        -------
        reduction_threshold : :py:class:`float` or :py:class:`None`
            :py:class:`None` if reduction of solution is not a criteria
        """
        if 'solution reduction' in self._conditions:
            return self._conditions['solution reduction']
        else:
            return None

    @property
    def min_residual(self):
        """Read-only accessor for the minimum residual threshold

        Returns
        -------
        residual_threshold : :py:class:`float` or :py:class:`None`
            :py:class:`None` if residual is not a criteria
        """
        if "residual" in self._conditions:
            return self._conditions["residual"]
        else:
            return None

    @property
    def min_error(self):
        """Read-only accessor for the minimum error threshold

        Returns
        -------
        error_threshold : :py:class:`float` or :py:class:`None`
            :py:class:`None` if error is not a criteria
        """
        if "error" in self._conditions:
            return self._conditions["error"]
        else:
            return None

    @property
    def max_iterations(self):
        """Read-only accessor for the maximum iterations threshold

        Returns
        -------
        iterations_threshold : :py:class:`int` or :py:class:`None`
            :py:class:`None` if iterations is not a criteria
        """
        if "iterations" in self._conditions:
            return self._conditions["iterations"]
        else:
            return None

    def print_conditions(self):
        """Pretty-formatted string of all active criteria and their thresholds
        """
        _outstr = ""
        first = True
        for cond in self._conditions:
            if not first:
                _outstr += ", "
            if cond in ThresholdCheck._default_min_conditions:
                _outstr += "{:s}={:.2e}".format(cond, self._conditions[cond])
            elif cond in ThresholdCheck._default_max_conditions:
                _outstr += "{:s}={:d}".format(cond, self._conditions[cond])
            first = False
        return _outstr

    def print_lines_for_log(self):
        _lines = OrderedDict()
        for _cond in self._conditions:
            if _cond in ThresholdCheck._default_min_conditions:
                _lines[_cond] = "{:.0e}".format(self._conditions[_cond])
            elif _cond in ThresholdCheck._default_max_conditions:
                _lines[_cond] = "{:d}".format(self._conditions[_cond])
        return _lines

    def compute_reduction(self, state):
        """Computes the reduction of the error and solution

        With respect to the supremum nomr of the given state's current iteration (see
        :py:attr:`.ISolverState.current_iteration` and :py:class:`.IIterationState`).
        In case no previous iteration is available, it immediatly returns.
        """
        if not state.previous_iteration:
            # there is no previous iteration to compare with
            LOG.debug("Skipping computation of reduction: No previous iteration available.")
            return

        if state.current_iteration.final_step.solution.error:
            # error is given; computing reduction of it
            _previous_error = supremum_norm(state.previous_iteration.final_step.solution.error)
            _current_error = supremum_norm(state.current_iteration.final_step.solution.error)
            state.solution.set_error_reduction(state.current_iteration_index,
                                               abs((_previous_error - _current_error) / _previous_error * 100))

        # computing reduction of solution
        _previous_solution = supremum_norm(state.previous_iteration.final_step.solution.value)
        _current_solution = supremum_norm(state.current_iteration.final_step.solution.value)
        state.solution.set_solution_reduction(state.current_iteration_index,
                                              abs((_previous_solution - _current_solution) / _previous_solution * 100))

    def _check_reduction(self, state):
        self.compute_reduction(state)
        if state.solution.error_reduction(state.current_iteration_index):
            self._check_minimum('error reduction', state.solution.error_reduction(state.current_iteration_index))
        if state.solution.solution_reduction(state.current_iteration_index):
            self._check_minimum('solution reduction', state.solution.solution_reduction(state.current_iteration_index))

    def _check_minimum(self, name, value):
        self._check("min", name, value)

    def _check_maximum(self, name, value):
        self._check("max", name, value)

    def _check(self, operator, name, value):
        _value = supremum_norm(value) if isinstance(value, (IDiagnosisValue, np.ndarray)) else value

        if name in self._conditions and self._conditions[name] is not None:
            assert_condition(_value is not None,
                             ValueError, "'{:s}' is a termination condition but not available to check."
                                         .format(name[0].capitalize() + name[1:]), self)

            if operator == "min":
                if _value <= self._conditions[name]:
                    LOG.debug("Minimum of {:s} reached: {:.2e} <= {:.2e}"
                              .format(name, _value, self._conditions[name]))
                    self._reason.append(name)
            elif operator == "max":
                if _value >= self._conditions[name]:
                    LOG.debug("Maximum of {:s} exceeded: {:d} >= {:d}"
                              .format(name, _value, self._conditions[name]))
                    self._reason.append(name)
            else:
                raise ValueError("Given operator '{:s}' is invalid.".format(operator))
        else:
            # $name is not a condition
            pass

    def _set_conditions(self, conditions):
        if isinstance(conditions, tuple):
            for cond in conditions:
                if cond in ThresholdCheck._default_min_conditions:
                    self._conditions[cond] = self._min_threshold
                elif cond in ThresholdCheck._default_max_conditions:
                    self._conditions[cond] = self._max_threshold
                else:
                    raise ValueError(func_name(self) +
                                     "Given condition is not supported: {:s}".format(cond))
        elif isinstance(conditions, dict):
            for cond in conditions:
                if cond in ThresholdCheck._default_min_conditions or \
                        cond in ThresholdCheck._default_max_conditions:
                    self._conditions[cond] = conditions[cond]
                else:
                    raise ValueError(func_name(self) +
                                     "Given condition is not supported: {:s}".format(cond))
        elif conditions is None:
            pass
        else:
            raise ValueError(func_name(self) +
                             "Given conditions can not be parsed: {:s}".format(conditions))

    def __str__(self):
        return "ThresholdCheck(" + self.print_conditions() + ")"
