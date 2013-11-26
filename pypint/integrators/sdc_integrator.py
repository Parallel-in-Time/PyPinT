# coding=utf-8

from .integrator_base import IntegratorBase
import numpy as np
from .node_providers.gauss_lobatto_nodes import GaussLobattoNodes
from .weight_function_providers.polynomial_weight_function import PolynomialWeightFunction
from pypint.utilities import func_name


class SdcIntegrator(IntegratorBase):
    """
    Summary
    -------
    Integral part of the SDC algorithm.
    """
    def __init__(self):
        super(self.__class__, self).__init__()
        self._smat = np.zeros(0)

    def init(self, nodes_type=GaussLobattoNodes(), num_nodes=3,
             weights_function=PolynomialWeightFunction(), interval=None):
        super(self.__class__, self).init(nodes_type, num_nodes, weights_function, interval)
        self._construct_s_matrix()

    def evaluate(self, data, **kwargs):
        """
        Extended Summary
        ----------------
        Computes the integral until the given node from the previous one.

        For integration nodes :math:`\\tau_i`, :math:`i=0,\\dots,n` specifying :math:`\\tau_3` as
        ``last_node_index`` results in the integral :math:`\\int_{\\tau_2}^{\\tau_3}`.

        Parameters
        ----------
        In addition to the options provided by :py:meth:`.IntegratorBase.evaluate` the following
        additional options are possible:

        last_node_index : integer
            (required)
            Index of the last node to integrate.

        Raises
        ------
        ValueError
            if ``until_node_index`` is not given

        See Also
        --------
        .IntegratorBase.evaluate
            overridden method
        """
        if "until_node_index" not in kwargs:
            raise ValueError(func_name(self) +
                             "Last node index must be given.")
        super(SdcIntegrator, self).evaluate(data, time_start=self.nodes[0],
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
