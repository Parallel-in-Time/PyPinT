# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class INodes(object):
    def __init__(self):
        self._num_nodes = None
        self._nodes = None

    def init(self, num_nodes):
        pass

    @property
    def nodes(self):
        return self._nodes

    @property
    def num_nodes(self):
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self._num_nodes = num_nodes
