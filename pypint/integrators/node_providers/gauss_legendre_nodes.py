# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_nodes import INodes
import numpy as np
from scipy import linalg
from pypint.utilities import *


class GaussLegendreNodes(INodes):
    """
    Summary
    -------
    Provider for Gauss-Legendre integration nodes with variable count.

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
        Initializes and computes Gauss-Legendre nodes.

        Parameters
        ----------
        n_nodes : integer
            The number of desired Gauss-Legendre nodes

        See Also
        --------
        .INodes.init
            overridden method
        """
        super().init(n_nodes, interval)
        self.num_nodes = n_nodes
        self._nodes = np.zeros(self.num_nodes)
        self._compute_nodes()
        self.interval = interval
        if interval is not None:
            super().transform()

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
        Accessor of number of Gauss-Legendre nodes.

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
                             "Gauss-Legendre with less than 2 nodes doesn't make any sense.")
        self._num_nodes = n_nodes

    def _compute_nodes(self):
        """
        Summary
        -------
        Computats nodes for the Gauss-Legendre quadrature of order :math:`n>1`
        on :math:`[-1,+1]`.

        Extended Summary
        ----------------
        (ported from MATLAB code, reference see below, original commend from
        MATLAB code:)

          Unlike many publicly available functions, this function is valid for
          :math:`n>=46`.
          This is due to the fact that it does not rely on MATLAB's build-in
          'root' routines to determine the roots of the Legendre polynomial, but
          finds the roots by looking for the eigenvalues of an alternative
          version of the companion matrix of the n'th degree Legendre
          polynomial.
          The companion matrix is constructed as a symmetrical matrix,
          guaranteeing that all the eigenvalues (roots) will be real.
          On the contrary, MATLAB's 'roots' function uses a general form for the
          companion matrix, which becomes unstable at higher orders :math:`n`,
          leading to complex roots.

        (Credit, where credit due)
        original MATLAB function by: Geert Van Damme <geert@vandamme-iliano.be>
        (February 21, 2010)
        """
        # Building the companion matrix comp_mat
        # comp_mat is such that det(nodes*I-comp_mat)=P_n(nodes), with P_n the
        # Legendre polynomial under consideration.
        # Moreover, comp_mat will be constructed in such a way that it is
        # symmetrical.
        linspace = np.linspace(start=1, stop=self.num_nodes - 1,
                        num=self.num_nodes - 1)
        a = linspace / np.sqrt(4.0 * linspace ** 2 - 1.0)
        comp_mat = np.diag(a, 1) + np.diag(a, -1)

        # Determining the abscissas (nodes)
        # - since det(nodesI-comp_mat)=P_n(nodes), the abscissas are the roots
        #   of the characteristic polynomial, i.d. the eigenvalues of comp_mat
        [eig_vals, eig_vecs] = linalg.eig(comp_mat)
        indizes = np.argsort(eig_vals)
        nodes = eig_vals[indizes]

        self._nodes = nodes.real