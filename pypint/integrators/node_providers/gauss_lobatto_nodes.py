# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_nodes import INodes
import numpy as np


class GaussLobattoNodes(INodes):
    """
    provider for Gauss-Lobatto integration nodes with variable count
    """
    def __init__(self):
        super().__init__()

    def init(self, n_nodes):
        """
        initializes and computes Gauss-Lobatto nodes

        :param n_nodes: number of nodes
        :type n_nodes:  integer

        :raises: see :py:func:`.num_nodes`
        """
        self.num_nodes = n_nodes
        self.__nodes = np.zeros(self.num_nodes)
        self._compute_nodes()

    @property
    def num_nodes(self):
        """
        number of nodes

        **Getter**

        :return: number of nodes
        :rtype:  integer

        **Setter**
        :param n_nodes: number of desired nodes
        :type n_nodes:  integer

        :raises: **ValueError** if ``n_nodes`` smaller 2
        """
        return self.__num_nodes

    @num_nodes.setter
    def num_nodes(self, n_nodes):
        if n_nodes < 2:
            raise ValueError(self.__qualname__ + ".init(): " +
                             "Gauss-Lobatto with less than 3 nodes doesn't make any sense.")
        self.__num_nodes = n_nodes

    def _compute_nodes(self):
        # TODO: Implement computation of Gauss-Lobatto nodes
        raise NotImplementedError(self.__qualname__ + "._compute_nodes(): " +
                                  "Computation of nodes not yet implemented.")
