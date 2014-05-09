# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np
from collections import OrderedDict

from pypint.problems.i_problem import IProblem
from pypint.utilities import assert_is_instance, assert_condition


class TransientProblemMixin(object):
    """Concept of a transient problem
    """
    def __init__(self, *args, **kwargs):
        self._time_start = kwargs.get('time_start', 0.0)
        self._time_end = kwargs.get('time_end', 1.0)

    @property
    def time_start(self):
        """Accessor for the time interval's start.

        Parameters
        ----------
        interval_start : :py:class:`float`
            Start point of the time interval.

        Returns
        -------
        interval_start : :py:class:`float`
            Start point of the time interval.

        Raises
        ------
        ValueError
            if given value is not a float
        """
        return self._time_start

    @time_start.setter
    def time_start(self, value):
        assert_is_instance(value, float, descriptor="Start Time", checking_obj=self)
        self._time_start = value

    @property
    def time_end(self):
        """Accessor for the time interval's end.

        Parameters
        ----------
        interval_end : :py:class:`float`
            End point of the time interval.

        Returns
        -------
        interval_end : :py:class:`float`
            End point of the time interval.

        Raises
        ------
        ValueError
            if given value is not a float
        """
        return self._time_end

    @time_end.setter
    def time_end(self, value):
        assert_is_instance(value, float, descriptor="End Time", checking_obj=self)
        self._time_end = value

    @property
    def time_interval(self):
        return np.array([self.time_start, self.time_end], dtype=np.float)

    @time_interval.setter
    def time_interval(self, value):
        assert_is_instance(value, np.ndarray, descriptor="Time Interval", checking_obj=self)
        assert_condition(value.size == 2, ValueError,
                         message="Time Interval must have two values: NOT %d" % value.size)
        assert_condition(value[0] < value[1], ValueError,
                         message="Start Time must be smaller than End Time: NOT %s >= %s" % (value[0], value[1]))
        self.time_start = value[0]
        self.time_end = value[1]

    def print_lines_for_log(self):
        return OrderedDict({'Time Interval': '[{:.3f}, {:.3f}]'.format(self.time_start, self.time_end)})

    def __str__(self):
        return r"t \in [{:.2f}, {:.2f}]".format(self.time_start, self.time_end)


def problem_is_transient(problem, checking_obj=None):
    assert_is_instance(problem, IProblem, message="It needs to be a problem to be time-dependet.",
                       checking_obj=checking_obj)
    return isinstance(problem, TransientProblemMixin)


__all__ = ['problem_is_transient', 'TransientProblemMixin']
