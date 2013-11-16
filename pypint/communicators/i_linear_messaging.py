# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.communicators.i_communication_provider import ICommunicationProvider


class ILinearMessaging(ICommunicationProvider):
    def __init__(self):
        super().__init__()
        self._compute_nodes = []

    def get_node(self, coordinate):
        return self._compute_nodes[coordinate]

    def get_next(self, node_id):
        return self._compute_nodes[node_id+1]

    def get_previous(self, node_id):
        return self._compute_nodes[node_id-1]
