# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class IParallelSolver(object):
    """basic interface for parallel solvers
    """

    def __init__(self):
        self._communicator = None

    @property
    def communicator(self):
        return self._communicator

    @communicator.setter
    def communicator(self, communicator):
        self._communicator = communicator
