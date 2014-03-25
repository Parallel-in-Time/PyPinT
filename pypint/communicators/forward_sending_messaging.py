# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.communicators.i_communication_provider import ICommunicationProvider
from pypint.utilities import assert_condition, assert_is_key, assert_is_instance, func_name
from pypint.utilities.logging import LOG


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
        LOG.debug(func_name(self, *args, **kwargs))
        self._next.write_buffer(*args, **kwargs)

    def receive(self, *args, **kwargs):
        """Returns this communicator's buffer

        Returns
        -------
        message : :py:class:`.Message`
        """
        super(ForwardSendingMessaging, self).receive(*args, **kwargs)
        LOG.debug(func_name(self) + str(self.buffer))
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
                         ValueError, "Exactly two communicators must be given: NOT %d" % len(kwargs),
                         self)

        assert_is_key(kwargs, 'previous', "Previous solver must be given.", self)
        assert_is_instance(kwargs['previous'], ForwardSendingMessaging,
                           "Previous Communicator must also be a ForwardSendingMessaging instance: NOT %s"
                           % kwargs['previous'].__class__.__name__,
                           self)
        self._previous = kwargs['previous']

        assert_is_key(kwargs, 'next', "Next solver must be given.", self)
        assert_is_instance(kwargs['next'], ForwardSendingMessaging,
                           "Next Communicator must also be a ForwardSendingMessaging instance: NOT %s"
                           % kwargs['next'].__class__.__name__,
                           self)
        self._next = kwargs['next']
