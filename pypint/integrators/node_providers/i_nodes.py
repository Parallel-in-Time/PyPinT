# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class INodes(object):
    def __init__(self):
        self.__num_nodes = None
        self.__nodes = None

    def init(self, num_nodes):
        pass

    @property
    def nodes(self):
        return self.__nodes

    @property
    def num_nodes(self):
        return self.__num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes = num_nodes
