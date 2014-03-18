# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from copy import deepcopy

from pypint.communicators import Message
from pypint.utilities import assert_is_instance, assert_condition


class ICommunicationProvider(object):
    """Interface for communication providers

    Notes
    -----
    This communication interface uses buffered communication.
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        buffer : :py:class:`.Message`
            initial buffer

        Raises
        ------
        ValueError
            if ``buffer`` is not a :py:class:`.Message`
        """
        self._buffer = None

        if 'buffer' in kwargs:
            assert_is_instance(kwargs['buffer'], Message,
                               "Initial buffer must be a Message: NOT %s" % kwargs['buffer'].__class__.__name__,
                               self)
            self._buffer = kwargs['buffer']
        else:
            self._buffer = Message()

    def send(self, *args, **kwargs):
        """Sending data to a specified communicator

        Notes
        -----
        Behaviour is implementation dependent.
        """
        pass

    def receive(self, *args, **kwargs):
        """Receiving data from a specified communicator

        Notes
        -----
        Behaviour is implementation dependent.
        """
        pass

    def link_solvers(self, *args, **kwargs):
        """Linking the specified communicators with this one

        Notes
        -----
        Behaviour is implementation dependent.
        """
        pass

    def write_buffer(self, *args, **kwargs):
        """Writes data into this communicator's buffer

        Parameters
        ----------
        value :
            data values to be send to the next solver
        time_point : :py:class:`float`
            time point of the data values
        flag : :py:class:`.Message.SolverFlag`
            message flag

        Raises
        ------
        ValueError

            * if no arguments are given
            * if ``time_point`` is not a :py:class:`float`
        """
        assert_condition(len(args) > 0 or len(kwargs) > 0,
                         ValueError, "At least one argument must be given: NOT %d (args) or %d (kwargs)"
                                     % (len(args), len(kwargs)),
                         self)

        if 'value' in kwargs:
            self._buffer.value = deepcopy(kwargs['value'])

        if 'time_point' in kwargs:
            assert_is_instance(kwargs['time_point'], float,
                               "Time point must be a float: NOT %s" % kwargs['time_point'].__class__.__name__,
                               self)
            self._buffer.time_point = deepcopy(kwargs['time_point'])

        if 'flag' in kwargs:
            self._buffer.flag = deepcopy(kwargs['flag'])

    @property
    def buffer(self):
        """Read-only accessor for this communicator's buffer

        Returns
        -------
        buffer : :py:class:`.Message`
        """
        return self._buffer
