# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class IntegratorBase(object):
    def __init__(self):
        self.__nodes = None
        self.__weights_function = None
        self.__weights = None

    def init(self, nodes_type, num_nodes, weights_function):
        pass

    def evaluate(self, data, time_start, time_end):
        pass

    @property
    def nodes(self):
        return self.__nodes.nodes

    @property
    def weights(self):
        return self.__weights
