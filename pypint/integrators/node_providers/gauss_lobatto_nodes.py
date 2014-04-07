# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np
import numpy.polynomial.legendre as leg

from pypint.integrators.node_providers.i_nodes import INodes
from pypint.utilities import assert_condition


class GaussLobattoNodes(INodes):
    """Provider for Gauss-Lobatto integration nodes with variable count.
    """

    std_interval = np.array([-1.0, 1.0])

    def __init__(self):
        super(GaussLobattoNodes, self).__init__()
        self._interval = GaussLobattoNodes.std_interval

    def init(self, n_nodes, interval=None):
        """Initializes and computes Gauss-Lobatto nodes.

        Parameters
        ----------
        n_nodes : :py:class:`int`
            The number of desired Gauss-Lobatto nodes

        See Also
        --------
        :py:meth:`.INodes.init` : overridden method
        """
        super(GaussLobattoNodes, self).init(n_nodes, interval)
        self.num_nodes = n_nodes
        self._nodes = np.zeros(self.num_nodes)
        self._compute_nodes()
        if interval is not None:
            self.transform(interval)

    @property
    def num_nodes(self):
        """Accessor of number of Gauss-Lobatto nodes.

        Raises
        ------
        ValueError
            If ``n_nodes`` is smaller than 2 *(only Setter)*.

        See Also
        --------
        :py:attr:`.INodes.num_nodes` : overridden method
        """
        return super(self.__class__, self.__class__).num_nodes.fget(self)

    @num_nodes.setter
    def num_nodes(self, n_nodes):
        super(self.__class__, self.__class__).num_nodes.fset(self, n_nodes)
        assert_condition(n_nodes >= 2,
                         ValueError, message="Fewer than 2 nodes do not make any sense.", checking_obj=self)
        self._num_nodes = n_nodes

    def _compute_nodes(self):
        """Computes Gauss-Lobatto integration nodes.

        Calculates the Gauss-Lobatto integration nodes via a root calculation of derivatives of the legendre
        polynomials.
        Note that the precision of float 64 is not guarantied.
        """
        roots = leg.legroots(leg.legder(np.array([0] * (self.num_nodes - 1) +
                                                 [1], dtype=np.float64)))
        self._nodes = np.array(np.append([-1.0], np.append(roots, [1.0])),
                               dtype=np.float64)
