# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from collections import OrderedDict

import numpy as np

from pypint.problems.i_problem import IProblem
from pypint.utilities import assert_is_instance, assert_condition, class_name


class TransientProblemMixin(object):
    """Concept of a transient problem
    """
    def __init__(self, *args, **kwargs):
        assert_is_instance(self, IProblem, descriptor="Problem needs to be a IProblem first: NOT %s" % class_name(self),
                           checking_obj=self)
        self.validate_time_interval(start=kwargs.get('time_start', 0.0), end=kwargs.get('time_end', 1.0))
        self._time_start = kwargs.get('time_start', 0.0)
        self._time_end = kwargs.get('time_end', 1.0)

    @property
    def time_start(self):
        """Accessor for the time interval's start.

        Parameters
        ----------
        interval_start : :py:class:`float`
            Start point of the time interval.
            Default=0.0

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
            Default=1.0

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
        self.validate_time_interval(start=value[0], end=value[1])
        self.time_start = value[0]
        self.time_end = value[1]

    def print_lines_for_log(self):
        return OrderedDict({'Time Interval': '[{:.3f}, {:.3f}]'.format(self.time_start, self.time_end)})

    def validate_time_interval(self, start=None, end=None):
        if start is None:
            start = self.time_start
        if end is None:
            end = self.time_end
        assert_condition(start < end, ValueError,
                         message="Start Time must be smaller than End Time: NOT %s >= %s" % (start, end),
                         checking_obj=self)
        return True

    def __str__(self):
        return r", t in [{:.2f}, {:.2f}]".format(self.time_start, self.time_end)


def problem_is_transient(problem, checking_obj=None):
    assert_is_instance(problem, IProblem, message="It needs to be a problem to be time-dependet.",
                       checking_obj=checking_obj)
    return isinstance(problem, TransientProblemMixin)


__all__ = ['problem_is_transient', 'TransientProblemMixin']
