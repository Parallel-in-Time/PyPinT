# coding=utf-8
"""
.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.communicators.i_communication_provider import ICommunicationProvider
from pypint.solvers.states.i_solver_state import ISolverState
from pypint.utilities import assert_is_key, assert_is_instance


class IParallelSolver(object):
    """basic interface for parallel solvers
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        communicator : :py:class:`.ICommunicationProvider`
        """
        assert_is_key(kwargs, 'communicator', "Communicator must be given", self)
        assert_is_instance(kwargs['communicator'], ICommunicationProvider,
                           "Communicator must be a valid communication provider: NOT %s"
                           % kwargs['communicator'].__class__.__name__,
                           self)
        self._communicator = kwargs['communicator']
        self._states = [ISolverState()]

    @property
    def comm(self):
        return self._communicator
