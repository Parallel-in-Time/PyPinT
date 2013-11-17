# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class INodes(object):
    """
    Summary
    -------
    Provider for integration nodes.

    Extended Summary
    ----------------
    This is an abstract interface for providers of integration nodes.
    """
    def __init__(self):
        self._num_nodes = None
        self._nodes = None

    def init(self, n_nodes):
        """
        Summary
        -------
        Initializes the vector of integration nodes of size `n_nodes`.

        Parameters
        ----------
        n_nodes : integer
            The number of desired integration nodes.

        Notes
        -----
        The implementation and behaviour must and will be defined by
        specializations of this interface.
        """
        pass

    @property
    def nodes(self):
        """
        Summary
        -------
        Accessor for the vector of integration nodes.

        Returns
        -------
        nodes : numpy.ndarray
            Vector of nodes.
        """
        return self._nodes

    @property
    def num_nodes(self):
        """
        Summary
        -------
        Accessor for the number of desired integration nodes.

        Returns
        -------
        number of nodes : integer
            The number of desired and/or computed integration nodes.

        Notes
        -----
        Specializations of this interface might override this accessor.
        """
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self._num_nodes = num_nodes
