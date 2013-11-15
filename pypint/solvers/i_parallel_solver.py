# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class IParallelSolver(object):
    """
    basic interface for parallel solvers
    """

    def __init__(self):
        self.__communicator = None

    @property
    def communicator(self):
        return self.__communicator

    @communicator.setter
    def communicator(self, communicator):
        self.__communicator = communicator
