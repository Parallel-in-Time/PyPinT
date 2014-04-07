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
        super(SdcIntegrator, self).__init__()
        self._smat = np.zeros(0)
        self._qmat = np.zeros(0)

    def init(self, nodes_type=GaussLobattoNodes, num_nodes=3, weights_function=PolynomialWeightFunction, interval=None):
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
        """Computes the integral until the given node from the previous one.

        For integration nodes :math:`\\tau_i`, :math:`i=1,\\dots,n` specifying :math:`\\tau_3` as ``target_node``
        results in the integral :math:`\\int_{\\tau_2}^{\\tau_3}`.

        Examples
        --------
        Given five integration nodes: :math:`\\tau_1, \\dots, \\tau_5`.

        To compute the integral from :math:`\\tau_2` to :math:`\\tau_3` one need to specify ``target_node`` as ``3`` and
        ``from_node`` as ``2``.
        Internally, the :math:`S`-matrix is used.

        To compute the full integral over all nodes one need to specify ``target_node`` as ``5`` only.
        Internally, the :math:`Q`-matrix is used.

        Parameters
        ----------
        target_node : :py:class:`int`
            *(required)*
            (1-based) index of the last node to integrate.

        from_node : :py:class:`int`
            *(optional)*
            (1-based) index of the first node to integrate from.
            *(defaults to ``0``)*

        Raises
        ------
        ValueError

            * if ``target_node`` is not given
            * if ``from_node`` is not smaller than ``target_node``

        See Also
        --------
        :py:meth:`.IntegratorBase.evaluate` : overridden method
        """
        assert_named_argument('target_node', kwargs, types=int, descriptor="Target Node Index", checking_obj=self)
        _target_index = kwargs["target_node"]

        _from_index = 0
        if 'from_node' in kwargs:
            assert_is_instance(kwargs['from_node'], int, descriptor="From Node Index", checking_obj=self)
            _from_index = kwargs['from_node']

        assert_condition(_from_index < _target_index,
                         ValueError,
                         message="Integration must cover at least two nodes: %d !< %d" % (_from_index, _target_index),
                         checking_obj=self)

        super(SdcIntegrator, self).evaluate(data, time_start=self.nodes[_from_index],
                                            time_end=self.nodes[_target_index])
        if _from_index != 0:
            assert_condition(_target_index <= self._smat.shape[0],
                             ValueError, message="Target Node Index {:d} too large. Must be within [{:d},{:d})"
                                                 .format(_target_index, 1, self._smat.shape[0]),
                             checking_obj=self)
            LOG.debug("Integrating from node {:d} to {:d} with S-Mat row {:d} on interval {}."
                      .format(_from_index, _target_index, _target_index - 1, self.nodes_type.interval))
            return np.dot(self._smat[_target_index - 1], data)
        else:
            assert_condition(_target_index < self._qmat.shape[0],
                             ValueError, message="Target Node Index {:d} too large. Must be within [{:d}, {:d}]"
                                                 .format(_target_index, 1, self._qmat.shape[0]),
                             checking_obj=self)
            LOG.debug("Integrating to node {:d} with Q-Mat row {:d} on interval {}."
                      .format(_target_index, _target_index, self.nodes_type.interval))
            return np.dot(self._qmat[_target_index], data)

    def transform_interval(self, interval):
        """Transforms nodes onto new interval

        See Also
        --------
        :py:meth:`.IntegratorBase.transform_interval` : overridden method
        """
        if interval is not None:
            if interval[0] - interval[-1] != self.nodes[0] - self.nodes[-1]:
                LOG.debug("Size of interval changed. Recalculating weights.")
                super(SdcIntegrator, self).transform_interval(interval)
                self._construct_s_matrix()
        else:
            # LOG.info("Cannot transform interval to None. Skipping.")
            pass
        # LOG.info("S-Matrix for interval {:s}:\n{:s}".format(interval, self._smat))
        # LOG.info("Q-Matrix for interval {:s}:\n{:s}".format(interval, self._qmat))

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

        # compute Q-matrix
        self._construct_q_matrix()

    def _construct_q_matrix(self):
        """Constructs integration :math:`Q`-matrix

        The :math:`Q`-matrix is the commulation of the rows of the :math:`S`-matrix.
        I.e. row :math:`i` of :math:`Q` is the sum of the rows :math:`0` to :math:`i - 1` of :math:`S`.

        However, :math:`Q` has one row more than :math:`S`, namely the first, which is constant zero.
        """
        self._qmat = np.zeros((self.nodes.size, self.nodes.size), dtype=float)
        for i in range(0, self._smat.shape[0]):
            self._qmat[i + 1] = self._qmat[i] + self._smat[i]

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
