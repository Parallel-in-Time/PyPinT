# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class ICommunicationProvider(object):
    """Interface for communication providers
    """
    def __init__(self):
        self._num_compute_nodes = None
        self._compute_nodes = None

    def get_neighbours(self, node_id):
        """Accessor for the neighbours of given node
        """
        pass

    def get_node(self, coordinate):
        """Accessor for a node of given coordinate
        """
        pass

    @property
    def num_compute_nodes(self):
        """Read-only accessor for the total number of nodes
        """
        return self._num_compute_nodes
