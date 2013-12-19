# coding=utf-8

from .integrator_base import IntegratorBase
from copy import deepcopy
import numpy as np
from .node_providers.gauss_lobatto_nodes import GaussLobattoNodes
from .weight_function_providers.polynomial_weight_function import PolynomialWeightFunction
from pypint.utilities import critical_assert
from pypint import LOG


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
            if ``last_node_index`` is not given

        See Also
        --------
        .IntegratorBase.evaluate
            overridden method
        """
        critical_assert("last_node_index" in kwargs, ValueError, "Last node index must be given.", self)
        _index = kwargs["last_node_index"]
        critical_assert(_index != 0 and _index <= self._smat.shape[0],
                        ValueError, "Last node index {:d} too small or too large. Must be within [{:d},{:d})"
                                    .format(_index, 1, self._smat.shape[0]),
                        self)
        super(SdcIntegrator, self).evaluate(data, time_start=self.nodes[0],
                                            time_end=self.nodes[_index])
        #LOG.debug("Integrating with S-Mat row {:d} ({:s}) on interval {:s}."
        #          .format(_index - 1, self._smat[_index - 1], self.nodes_type.interval))
        return np.dot(self._smat[_index - 1], data)

    def transform_interval(self, interval):
        if interval is not None:
            if interval[0] - interval[-1] != self.nodes[0] - self.nodes[-1]:
                #LOG.debug("Size of interval changed. Recalculating weights.")
                super(SdcIntegrator, self).transform_interval(interval)
                self._construct_s_matrix()
                #LOG.debug("S-Matrix for interval {:s}: {:s}".format(interval, self._smat))
        else:
            LOG.debug("Cannot transform interval to None.")
            pass

    def _construct_s_matrix(self):
        critical_assert(isinstance(self._nodes, GaussLobattoNodes),
                        ValueError, "Other than Gauss-Lobatto integration nodes not yet supported.",
                        self)
        self._smat = np.zeros((self.nodes.size - 1, self.nodes.size), dtype=float)
        for i in range(1, self.nodes.size):
            self.weights_function.evaluate(self.nodes, np.array([self.nodes[i - 1], self.nodes[i]]))
            self._smat[i - 1] = self.weights_function.weights

    def __copy__(self):
        copy = self.__class__.__new__(self.__class__)
        copy.__dict__.update(self.__dict__)
        return copy

    def __deepcopy__(self, memo):
        copy = self.__class__.__new__(self.__class__)
        memo[id(self)] = copy
        for item, value in self.__dict__.items():
            setattr(copy, item, deepcopy(value, memo))
        return copy
