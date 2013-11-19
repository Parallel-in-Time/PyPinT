# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class ICommunicationProvider(object):
    def __init__(self):
        self._num_compute_nodes = None
        self._compute_nodes = None

    def get_neighbours(self, node_id):
        pass

    def get_node(self, coordinate):
        pass

    @property
    def num_compute_nodes(self):
        return self._num_compute_nodes
