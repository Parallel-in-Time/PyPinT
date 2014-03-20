# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from enum import Enum, unique
from pypint.utilities import assert_is_instance


class Message(object):
    """Container for inter-solver messages
    """

    @unique
    class SolverFlag(Enum):
        """State flags of the sending solver
        """

        #: no flag set
        none = 0

        #: still iterating
        iterating = 1

        #: finished and converged with respect to given thresholds
        converged = 10

        #: finished due to other reasons than thresholds or errors
        finished = 11

        #: finished due to an error
        failed = 30

        #: width of the interval have been changed (i.e. any subsequent intervals need to be recomputed)
        time_adjusted = 40

    def __init__(self):
        self._value = None
        self._time_point = None
        self._flag = Message.SolverFlag.none

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @value.deleter
    def value(self):
        self._value = None

    @property
    def time_point(self):
        return self._time_point

    @time_point.setter
    def time_point(self, time_point):
        self._time_point = time_point

    @time_point.deleter
    def time_point(self):
        self._time_point = None

    @property
    def flag(self):
        return self._flag

    @flag.setter
    def flag(self, flag):
        assert_is_instance(flag, Message.SolverFlag,
                           "Given flag is not a SolverFlag: %s" % type(flag))
        self._flag = flag

    @flag.deleter
    def flag(self):
        self._flag = Message.SolverFlag.none

    def __str__(self):
        return "Message(value=%s, time_point=%s, flag=%s)" % (self.value, self.time_point, self.flag)
