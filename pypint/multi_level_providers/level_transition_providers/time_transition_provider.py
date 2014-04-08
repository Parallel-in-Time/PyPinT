# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np

from pypint.multi_level_providers.level_transition_providers.i_level_transition_provider import ILevelTransitionProvider
from pypint.utilities import assert_named_argument, assert_condition


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
        assert_condition(len(kwargs['fine_nodes'].shape) == 0, ValueError,
                         message="Fine Nodes must have a single dimension: NOT %s" % kwargs['fine_nodes'].shape,
                         checking_obj=self)

        assert_named_argument('coarse_nodes', kwargs, np.ndarray, descriptor='Coarse Nodes', checking_obj=self)
        assert_condition(len(kwargs['coarse_nodes'].shape) == 0, ValueError,
                         message="Coarse Nodes must have a single dimension: NOT %s" % kwargs['coarse_nodes'].shape,
                         checking_obj=self)

        assert_condition(kwargs['fine_nodes'].size > kwargs['coarse_nodes'].size, ValueError,
                         message="There must be more Fine Nodes than Coarse Nodes: NOT %d <= %d"
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
        return np.dot(self.prolongation_operator, coarse_data)

    def restringate(self, fine_data):
        super(TimeTransitionProvider, self).restringate(fine_data)
        return np.dot(self.restringation_operator, fine_data)

    def _compute_prolongation_matrix(self):
        self._prolongation_operator = np.zeros((self.num_fine_points, self.num_coarse_points), dtype=float)
        for k in range(0, self.num_fine_points):
            for j in range(0, self.num_coarse_points):
                self._prolongation_operator[k][j] = self._lagrange_polynome(j, self._coarse_nodes, self._fine_nodes[k])
        #     self._prolongation_operator[k] = self._scaled_interpolation_weights(from_points=self._coarse_nodes,
        #                                                                         to_points=self._fine_nodes,
        #                                                                         index=k)

    def _compute_restringation_matrix(self):
        self._restringation_operator = np.zeros((self.num_coarse_points, self.num_fine_points), dtype=float)
        for k in range(0, self.num_coarse_points):
            for j in range(0, self.num_fine_points):
                self._restringation_operator[k][j] = self._lagrange_polynome(j, self._fine_nodes, self._coarse_nodes[k])
        #     self._restringation_operator[k] = self._scaled_interpolation_weights(from_points=self._fine_nodes,
        #                                                                          to_points=self._coarse_nodes,
        #                                                                          index=k)

    @staticmethod
    def _lagrange_polynome(j, base_points, x):
        """Evaluates :math:`j`th Lagrange polynomial based on ``base_points`` at :math:`x`

        For a given set of :math:`n` nodes :math:`\\vec{b}` (``base_points``) the :math:`j`th Lagrange polynomial is
        constructed and evaluated at the given point :math:`x`.

        .. math::

            P_j(x) = \\prod_{m=1, m \\neq j}^{n} \frac{x - b_m}{b_j - b_m}

        Parameters
        ----------
        j : :py:class:`int`
        base_points : :py:class:`numpy.ndarray` of :math:`n` :py:class:`float`
            points to construct the Lagrange polynome on
        x : :py:class:`float`
            point to evaluate the Lagrange polynome at

        Returns
        -------
        value : :py:class:`float`
            value of the specified Lagrange polynome
        """
        _val = 1.0
        for m in range(0, base_points.size):
            if m != j:
                _val *= (x - base_points[m]) / (base_points[j] - base_points[m])
        return _val

    def _scaled_interpolation_weights(self, from_points, to_points, index):
        _weights = np.zeros(from_points.size, dtype=float)
        if to_points[index] in from_points:
            _weights[from_points.tolist().index(to_points[index])] = 1.0
        else:
            _width = abs(to_points[-1] - to_points[0])
            for j in range(0, from_points.size):
                _weights[j] = pow(1 - abs(to_points[index] - from_points[j]) / _width, self._weight_scale)
            _weights /= np.sum(_weights)

        return _weights
