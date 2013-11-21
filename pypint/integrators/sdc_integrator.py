# coding=utf-8

from .integrator_base import IntegratorBase
import numpy as np
from .node_providers.gauss_lobatto_nodes import GaussLobattoNodes
from .weight_function_providers.polynomial_weight_function import PolynomialWeightFunction
from pypint.utilities import func_name


class SdcIntegrator(IntegratorBase):
    def __init__(self):
        super(self.__class__, self).__init__()
        self._smat = np.zeros(0)

    def init(self, nodes_type=GaussLobattoNodes(), num_nodes=3,
             weights_function=PolynomialWeightFunction(), interval=None):
        super(self.__class__, self).init(nodes_type, num_nodes, weights_function, interval)
        self._construct_s_matrix()

    def evaluate(self, data, **kwargs):
        """
        Raises
        ------
        ValueError
            if ``until_node_index`` is not given
        """
        if "until_node_index" not in kwargs:
            raise ValueError(func_name(self) +
                             "Last node index must be given.")
        super(self.__class__, self).evaluate(data, time_start=self.nodes[0],
                                             time_end=self.nodes[kwargs["until_node_index"] + 1])
        return np.dot(self._smat[kwargs["until_node_index"]], data)

    def _construct_s_matrix(self):
        if isinstance(self._nodes, GaussLobattoNodes):
            self._smat = np.zeros((self.nodes.size - 1, self.nodes.size), dtype=float)
            for i in range(1, self.nodes.size):
                self.weights_function.evaluate(self.nodes, np.array([self.nodes[i - 1], self.nodes[i]]))
                self._smat[i - 1] = self.weights_function.weights
        else:
            raise ValueError(func_name(self) +
                             "Other than Gauss-Lobatto integration nodes not yet supported.")
