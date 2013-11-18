# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_nodes import INodes
import numpy as np


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
    def __init__(self):
        super().__init__()

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
        pypint.integrators.node_providers.i_nodes.INodes.init
            overridden method
        """
        self.num_nodes = n_nodes
        self._nodes = np.zeros(self.num_nodes)
        self._compute_nodes()
        self.interval = interval
        if interval is not None:
            self.transform(self.interval)

    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, interval):
        if interval is None:
            self._interval = np.array([-1.0, 1.0])
        elif not isinstance(interval, np.ndarray) or interval.size != 2:
            ValueError("")
        else:
            self._interval = interval

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
        pypint.integrators.node_providers.i_nodes.INodes.num_nodes
            overridden method
        """
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, n_nodes):
        if n_nodes < 2:
            raise ValueError(self.__qualname__ + ".init(): " +
                             "Gauss-Lobatto with less than 3 nodes doesn't make any sense.")
        self._num_nodes = n_nodes

    def _compute_nodes(self):
        # TODO: Implement computation of Gauss-Lobatto nodes
        raise NotImplementedError(self.__qualname__ + "._compute_nodes(): " +
                                  "Computation of nodes not yet implemented.")
