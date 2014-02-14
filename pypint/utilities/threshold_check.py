# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np

from pypint.solvers.diagnosis import IDiagnosisValue
from pypint.solvers.diagnosis.norms import supremum_norm
from pypint.utilities import assert_condition, func_name
from pypint import LOG


class ThresholdCheck(object):
    _default_min_threshold = 1e-7
    _default_max_threshold = 10
    _default_min_conditions = {
        "reduction": _default_min_threshold,
        "residual": _default_min_threshold,
        "error": _default_min_threshold
    }
    _default_max_conditions = {
        "iterations": _default_max_threshold
    }

    def __init__(self, min_threshold=_default_min_threshold, max_threshold=_default_max_threshold,
                 conditions=("residual", "iterations")):
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._conditions = {}
        self._set_conditions(conditions)
        self._reason = None

    def check(self, state):
        self._check_reduction(state)
        self._check_minimum("residual", state.current_iteration.final_step.solution.residual)
        self._check_minimum("error", state.current_iteration.final_step.solution.error)
        self._check_maximum("iterations", state.current_iteration_index + 1)

    def has_reached(self, human=False):
        if human:
            return "Threshold condition met: {:s}".format(self._reason)
        else:
            return self._reason

    @property
    def min_reduction(self):
        if "reduction" in self._conditions:
            return self._conditions["reduction"]
        else:
            return None

    @property
    def min_residual(self):
        if "residual" in self._conditions:
            return self._conditions["residual"]
        else:
            return None

    @property
    def min_error(self):
        if "error" in self._conditions:
            return self._conditions["error"]
        else:
            return None

    @property
    def max_iterations(self):
        if "iterations" in self._conditions:
            return self._conditions["iterations"]
        else:
            return None

    def print_conditions(self):
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

    def compute_reduction(self, state):
        """
        Summary
        -------
        Computes the reduction of the error and solution with respect to the supremum nomr of the given state's current
        iteration (see :py:attr:`.ISolverState.current_iteration` and :py:class:`.IIterationState`).
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
                                              abs((_previous_solution - _current_solution) / _current_solution * 100))

    def _check_reduction(self, state):
        self.compute_reduction(state)

        if state.solution.error_reduction(state.current_iteration_index):
            self._check_minimum("reduction", state.solution.error_reduction(state.current_iteration_index))
        elif state.solution.solution_reduction(state.current_iteration_index):
            self._check_minimum("reduction", state.solution.solution_reduction(state.current_iteration_index))
        else:
            # no reduction availbale
            pass

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
                    self._reason = name
            elif operator == "max":
                if _value >= self._conditions[name]:
                    LOG.debug("Maximum of {:s} exceeded: {:d} > {:d}"
                              .format(name, _value, self._conditions[name]))
                    self._reason = name
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
