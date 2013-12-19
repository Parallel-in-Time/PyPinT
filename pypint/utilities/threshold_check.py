# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.utilities import critical_assert, func_name
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

    def check(self, reduction=None, residual=None, error=None, iterations=None):
        self._check_minimum("reduction", reduction)
        self._check_minimum("residual", residual)
        self._check_minimum("error", error)
        self._check_maximum("iterations", iterations)

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

    def _check_minimum(self, name, value):
        self._check("min", name, value)

    def _check_maximum(self, name, value):
        self._check("max", name, value)

    def _check(self, operator, name, value):
        if name in self._conditions and self._conditions[name] is not None:
            critical_assert(value is not None,
                            ValueError, "'{:s}' is a termination condition but not available to check."
                                        .format(name[0].capitalize() + name[1:]), self)

            if operator == "min":
                if value <= self._conditions[name]:
                    LOG.debug("Minimum of {:s} reached: {:.2e} <= {:.2e}"
                              .format(name, value, self._conditions[name]))
                    self._reason = name
            elif operator == "max":
                if value > self._conditions[name]:
                    LOG.debug("Maximum of {:s} exceeded: {:d} > {:d}"
                              .format(name, value, self._conditions[name]))
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

    def print_conditions(self):
        str = ""
        first = True
        for cond in self._conditions:
            if not first:
                str += ", "
            if cond in ThresholdCheck._default_min_conditions:
                str += "{:s}={:.2e}".format(cond, self._conditions[cond])
            elif cond in ThresholdCheck._default_max_conditions:
                str += "{:s}={:d}".format(cond, self._conditions[cond])
            first = False
        return str

    def __str__(self):
        return "ThresholdCheck(" + self.print() + ")"
