# coding=utf-8
"""
.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.communicators.i_communication_provider import ICommunicationProvider
from pypint.utilities import assert_named_argument


class IParallelSolver(object):
    """basic interface for parallel solvers
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        communicator : :py:class:`.ICommunicationProvider`
        """
        assert_named_argument('communicator', kwargs, types=ICommunicationProvider, descriptor="Communicator",
                              checking_obj=self)
        self._communicator = kwargs['communicator']
        self._states = []

    @property
    def comm(self):
        return self._communicator
