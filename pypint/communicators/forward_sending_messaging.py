# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.communicators.i_communication_provider import ICommunicationProvider
from pypint.utilities import assert_condition, assert_named_argument
from pypint.utilities.logging import this_got_called


class ForwardSendingMessaging(ICommunicationProvider):
    """A linear forward-directed communication pattern
    """
    def __init__(self, *args, **kwargs):
        super(ForwardSendingMessaging, self).__init__(*args, **kwargs)
        self._previous = None
        self._next = None

    def send(self, *args, **kwargs):
        """Sends given message to the next communicator

        See Also
        --------
        :py:meth:`.ICommunicationProvider.write_buffer`
            for allowed arguments
        """
        super(ForwardSendingMessaging, self).send(*args, **kwargs)
        # this_got_called(self, *args, **kwargs)
        self._next.write_buffer(*args, **kwargs)

    def receive(self, *args, **kwargs):
        """Returns this communicator's buffer

        Returns
        -------
        message : :py:class:`.Message`
        """
        super(ForwardSendingMessaging, self).receive(*args, **kwargs)
        if 'tag' in kwargs:
            # this_got_called(self, *args, add_log_msg=str(self.tagged_buffer(tag=kwargs['tag'])), **kwargs)
            return self.tagged_buffer(tag=kwargs['tag'])
        else:
            # this_got_called(self, *args, add_log_msg=str(self.buffer), **kwargs)
            return self.buffer

    def link_solvers(self, *args, **kwargs):
        """Links the given communicators with this communicator

        Parameters
        ----------
        previous : :py:class:`.ForwardSendingMessaging`
            communicator of the previous solver
        next : :py:class:`.ForwardSendingMessaging`
            communicator of the next solver

        Raises
        ------
        ValueError
            if one of the two communicators of the specified type is not given
        """
        super(ForwardSendingMessaging, self).link_solvers(*args, **kwargs)
        assert_condition(len(kwargs) == 2,
                         ValueError, message="Exactly two communicators must be given: NOT %d" % len(kwargs),
                         checking_obj=self)

        assert_named_argument('previous', kwargs, types=ForwardSendingMessaging, descriptor="Previous Communicator",
                              checking_obj=self)
        self._previous = kwargs['previous']

        assert_named_argument('next', kwargs, types=ForwardSendingMessaging, descriptor="Next Communicator",
                              checking_obj=self)
        self._next = kwargs['next']
