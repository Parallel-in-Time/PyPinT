# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from copy import deepcopy

import numpy as np

from pypint.integrators.integrator_base import IntegratorBase
from pypint.integrators.node_providers.gauss_lobatto_nodes import GaussLobattoNodes
from pypint.integrators.weight_function_providers.polynomial_weight_function import PolynomialWeightFunction
from pypint.utilities import assert_is_instance, assert_condition, assert_named_argument
from pypint.utilities.logging import LOG


class SdcIntegrator(IntegratorBase):
    """Integral part of the SDC algorithm.
    """
    def __init__(self):
        super(self.__class__, self).__init__()
        self._smat = np.zeros(0)

    def init(self, nodes_type=GaussLobattoNodes(), num_nodes=3, weights_function=PolynomialWeightFunction(),
             interval=None):
        """Initialize SDC Integrator

        Parameters
        ----------
        nodes_type : :py:class:`.INodes`
            type of the nodes
            (defaults to :py:class:`.GaussLobattoNodes`)

        num_nodes : :py:class:`int`
            number of nodes
            (defaults to 3)

        weights_function : :py:class:`.IWeightFunction`
            type of the weights function
            (defaults to :py:class:`.PolynomialWeightFunction`)

        interval : :py:class:`numpy.ndarray` or :py:class:`None`
            interval for the nodes
            (see :py:meth:`.INodes.transform` for possible values)
        """
        super(SdcIntegrator, self).init(nodes_type, num_nodes, weights_function, interval)
        self._construct_s_matrix()

    def evaluate(self, data, **kwargs):
        """

        Computes the integral until the given node from the previous one.

        For integration nodes :math:`\\tau_i`, :math:`i=0,\\dots,n` specifying :math:`\\tau_3` as ``last_node_index``
        results in the integral :math:`\\int_{\\tau_2}^{\\tau_3}`.

        Parameters
        ----------
        last_node_index : :py:class:`int`
            (required)
            Index of the last node to integrate.

        Raises
        ------
        ValueError
            if ``last_node_index`` is not given

        See Also
        --------
        :py:meth:`.IntegratorBase.evaluate` : overridden method
        """
        assert_named_argument('last_node_index', kwargs, types=int, descriptor="Last Node Index", checking_obj=self)
        _index = kwargs["last_node_index"]
        assert_condition(_index != 0 and _index <= self._smat.shape[0],
                         ValueError, message="Last node index {:d} too small or too large. Must be within [{:d},{:d})"
                                             .format(_index, 1, self._smat.shape[0]),
                         checking_obj=self)
        super(SdcIntegrator, self).evaluate(data, time_start=self.nodes[0],
                                            time_end=self.nodes[_index])
        # LOG.debug("Integrating {:s} with S-Mat row {:d} ({:s}) on interval {:s}."
        #           .format(data, _index - 1, self._smat[_index - 1], self.nodes_type.interval))
        return np.dot(self._smat[_index - 1], data)

    def transform_interval(self, interval):
        """Transforms nodes onto new interval

        See Also
        --------
        :py:meth:`.IntegratorBase.transform_interval` : overridden method
        """
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
        """Constructs integration :math:`S`-matrix

        Rows of the matrix are the integration from one node to the next.
        I.e. row :math:`i` integrates from node :math:`i-1` to node :math:`i`.
        """
        assert_is_instance(self._nodes, GaussLobattoNodes,
                           message="Other than Gauss-Lobatto integration nodes not yet supported.", checking_obj=self)
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
