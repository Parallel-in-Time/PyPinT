# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class INodes(object):
    def __init__(self):
        self._nodes = None

    def init(self, num_nodes):
        pass

    @property
    def nodes(self):
        return self._nodes
