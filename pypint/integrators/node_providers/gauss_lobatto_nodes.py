# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_nodes import INodes
import numpy as np
import numpy.polynomial.legendre as leg
from pypint.utilities import func_name


class GaussLobattoNodes(INodes):
    """
    Summary
    -------
    Provider for Gauss-Lobatto integration nodes with variable count.

    Extended Summary
    ----------------

    Examples
    --------
    """

    std_interval = np.array([-1.0, 1.0])

    def __init__(self):
        super(GaussLobattoNodes, self).__init__()

    def init(self, n_nodes, interval=None):
        """
        Summary
        -------
        Initializes and computes Gauss-Lobatto nodes.

        Parameters
        ----------
        n_nodes : integer
            The number of desired Gauss-Lobatto nodes

        See Also
        --------
        .INodes.init
            overridden method
        """
        super(GaussLobattoNodes, self).init(n_nodes, interval)
        self.num_nodes = n_nodes
        self._nodes = np.zeros(self.num_nodes)
        self._compute_nodes()
        self.interval = interval
        if interval is not None:
            self.transform(interval)

    @property
    def interval(self):
        """
        Summary
        -------
        Accessor for integration nodes interval.

        Extended Summary
        ----------------
        Default nodes interval for Gauss integration is :math:`[-1,1]`.

        See Also
        --------
        .INodes.interval
            overridden accessor
        """
        return super(self.__class__, self.__class__).interval.fget(self)

    @interval.setter
    def interval(self, interval):
        if interval is None:
            self._interval = np.array([-1.0, 1.0])
        super(self.__class__, self.__class__).interval.fset(self, interval)

    @property
    def num_nodes(self):
        """
        Summary
        -------
        Accessor of number of Gauss-Lobatto nodes.

        Raises
        ------
        ValueError
            If ``n_nodes`` is smaller than 2 *(only Setter)*.

        See Also
        --------
        .INodes.num_nodes
            overridden method
        """
        return super(self.__class__, self.__class__).num_nodes.fget(self)

    @num_nodes.setter
    def num_nodes(self, n_nodes):
        super(self.__class__, self.__class__).num_nodes.fset(self, n_nodes)
        if n_nodes < 2:
            raise ValueError(func_name(self) +
                             "Fewer than 2 nodes do not make any sense.")
        self._num_nodes = n_nodes

    def _compute_nodes(self):
        """
        Summary
        -------
        Computes Gauss-Lobatto integration nodes.

        Extended Summary
        ----------------
        Calculates the Gauss-Lobatto integration nodes via a root calculation
        of derivatives of the legendre polynomials.
        Note that the precision of float 64 is not guarantied.
        """
        roots = leg.legroots(leg.legder(np.array([0] * (self.num_nodes - 1) +
                                                 [1], dtype=np.float64)))
        self._nodes = np.array(np.append([-1.0], np.append(roots, [1.0])),
                               dtype=np.float64)
