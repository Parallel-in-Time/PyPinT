# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class IntegratorBase(object):
    def __init__(self):
        self._nodes = None
        self._weights_function = None
        self._weights = None

    def init(self, nodes_type, num_nodes, weights_function):
        pass

    def evaluate(self, data, time_start, time_end):
        pass

    @property
    def nodes(self):
        return self._nodes.nodes

    @property
    def weights(self):
        return self._weights
