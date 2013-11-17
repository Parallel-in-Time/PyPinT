# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class IWeightFunction(object):
    def __init__(self):
        self._weights = None

    def init(self, function):
        pass

    def evaluate(self, nodes):
        """
        computes weights for given nodes based on set polynomial weight function

        :param nodes: array of nodes to compute weights for
        :type nodes:  numpy.ndarray
        :returns:     computed weights
        :rtype:       numpy.ndarray
        """
        pass

    @property
    def weights(self):
        """
        accessor for computed weights

        :return: computed weights
        :rtype:  numpy.ndarray
        """
        return self._weights
