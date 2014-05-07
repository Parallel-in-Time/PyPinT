# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np
from math import fabs

from pypint.multi_level_providers.level_transition_providers.i_level_transition_provider import ILevelTransitionProvider
from pypint.utilities.math import lagrange_polynome
from pypint.utilities import assert_named_argument, assert_condition
from pypint.utilities.logging import LOG


class TimeTransitionProvider(ILevelTransitionProvider):
    """Level Transition Provider between two time levels

    Provides prolongation and restringation between two time levels based on interpolation.

    In case of nested notes (i.e. fine level with 5 Gauss-Lobatto nodes, coarse level with 3 Gauss-Lobatto nodes)
    the restringation is a simple injection.

    Prolongation is always done via plain interpolation.
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        fine_nodes : :py:class:`numpy.ndarray` of :py:class:`float`
        coarse_nodes : :py:class:`numpy.ndarray` of :py:class:`float`
        """
        super(TimeTransitionProvider, self).__init__(*args, **kwargs)
        assert_named_argument('fine_nodes', kwargs, np.ndarray, descriptor='Fine Nodes', checking_obj=self)
        assert_condition(len(kwargs['fine_nodes'].shape) == 1, ValueError,
                         message="Fine Nodes must have a single dimension: NOT %s" % kwargs['fine_nodes'].shape,
                         checking_obj=self)

        assert_named_argument('coarse_nodes', kwargs, np.ndarray, descriptor='Coarse Nodes', checking_obj=self)
        assert_condition(len(kwargs['coarse_nodes'].shape) == 1, ValueError,
                         message="Coarse Nodes must have a single dimension: NOT %s" % kwargs['coarse_nodes'].shape,
                         checking_obj=self)

        assert_condition(kwargs['fine_nodes'].size >= kwargs['coarse_nodes'].size, ValueError,
                         message="There must be more or at least as many Fine Nodes than Coarse Nodes: NOT %d < %d"
                                 % (kwargs['fine_nodes'].size, kwargs['coarse_nodes'].size),
                         checking_obj=self)

        self._n_fine_points = kwargs['fine_nodes'].size
        self._fine_nodes = kwargs['fine_nodes']

        self._n_coarse_points = kwargs['coarse_nodes'].size
        self._coarse_nodes = kwargs['coarse_nodes']

        self._weight_scale = kwargs['weight_scale'] if 'weight_scale' in kwargs else 2

        self._compute_prolongation_matrix()
        self._compute_restringation_matrix()

    def prolongate(self, coarse_data):
        super(TimeTransitionProvider, self).prolongate(coarse_data)
        return np.tensordot(self.prolongation_operator, coarse_data, axes=([1], [0]))
        # return np.dot(self.prolongation_operator, coarse_data)

    def restringate(self, fine_data):
        super(TimeTransitionProvider, self).restringate(fine_data)
        return np.tensordot(self.restringation_operator, fine_data, axes=([1], [0]))
        # return np.dot(self.restringation_operator, fine_data)

    def _compute_prolongation_matrix(self):
        self._prolongation_operator = np.zeros((self.num_fine_points, self.num_coarse_points), dtype=float)
        for k in range(0, self.num_fine_points):
            for j in range(0, self.num_coarse_points):
                self._prolongation_operator[k][j] = lagrange_polynome(j, self._coarse_nodes, self._fine_nodes[k])
        #     self._prolongation_operator[k] = self._scaled_interpolation_weights(from_points=self._coarse_nodes,
        #                                                                         to_points=self._fine_nodes,
        #                                                                         index=k)
        # LOG.debug("Prolongation Operator: %s" % self._prolongation_operator)

    def _compute_restringation_matrix(self):
        self._restringation_operator = np.zeros((self.num_coarse_points, self.num_fine_points), dtype=float)
        for k in range(0, self.num_coarse_points):
            for j in range(0, self.num_fine_points):
                self._restringation_operator[k][j] = lagrange_polynome(j, self._fine_nodes, self._coarse_nodes[k])
                #     self._restringation_operator[k] = self._scaled_interpolation_weights(from_points=self._fine_nodes,
                #                                                                          to_points=self._coarse_nodes,
                #                                                                          index=k)
        # LOG.debug("Restringation Operator: %s" % self._restringation_operator)

    def _scaled_interpolation_weights(self, from_points, to_points, index):
        """non-proofen custom integration method

        This computes the integration weights based on their relative distance.
        The distance contributes squared to weaken nodes further away in contrast to nodes close by.

        For two given sets of nodes :math:`\\vec{x}^C \\in \\mathbb{R}^n` and :math:`\\vec{x}^F \\in \\mathbb{R}^N`
        with :math:`n < N` the interpolation weights for the interpolation from :math:`\\vec{x}^C` onto
        :math:`\\vec{x}^F` at a target node :math:`x^F \\in \\vec{x}^F`:

        .. math::

            \\tilde{\\omega}_i = \\left( 1 - \\frac{|x^F - x_i^C|}{|x_0 - x_N|} \\right) ^ 2

        To counter unwanted scaling effects, the computed weights are scaled to sum up to :math:`1`:

        .. math::

            \\vec{\\omega} = \\frac{\\tilde{\\omega}}{\\sum_{i=1}^n \\tilde{\\omega}_i}

        Parameters
        ----------
        from_points : :py:class:`numpy.ndarray` of :math:`n` :py:class:`float`
            nodes to interpolate from; the Lagrange polynome is based on these
        to_points : :py:class:`numpy.ndarray` of :math:`N` :py:class:`float`
            nodes to interpolate onto; the Lagrange polynome is evaluated at one of these
        index : :py:class:`float`
            index of the target node to evaluate the Lagrange polynome at

        Returns
        -------
        value : :py:class:`float`
            interpolation weight for given nodes

        Notes
        -----
        Please be aware, that this method has not been mathematically or numerically tested and validated.
        """
        _weights = np.zeros(from_points.size, dtype=float)
        _width = abs(to_points[-1] - to_points[0])
        for j in range(0, from_points.size):
            _weights[j] = pow(1 - abs(to_points[index] - from_points[j]) / _width, self._weight_scale)
        _weights /= np.sum(_weights)

        return _weights

    def __str__(self):
        return "TimeTransitionProvider<0x%x>(fine_nodes=%d, coarse_nodes=%d)" \
               % (id(self), self._n_fine_points, self._n_coarse_points)
