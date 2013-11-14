# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.plugins.loggers.logger_base import *


class ICommunicationProvider(object):
    def __init__(self):
        self.__num_compute_nodes = None
        self.__compute_nodes = None
        self.__logger = LoggerBase

    def get_neighbours(self, node_id):
        pass

    def get_node(self, coordinate):
        pass

    @property
    def num_compute_nodes(self):
        return self.__num_compute_nodes
