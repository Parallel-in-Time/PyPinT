# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_communication_provider import *


class ILinearMessaging(ICommunicationProvider):
    def __init__(self):
        self.__compute_nodes = []
        super().__init__()

    def get_node(self, coordinate):
        return self.__compute_nodes[node_id]

    def get_next(self, node_id):
        return self.__compute_nodes[node_id+1]

    def get_previous(self, node_id):
        return self.__compute_nodes[node_id-1]
