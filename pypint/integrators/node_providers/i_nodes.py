# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

import numpy as np


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
        self._interval = None

    def init(self, n_nodes, interval=None):
        """
        Summary
        -------
        Initializes the vector of integration nodes of size `n_nodes`.

        Parameters
        ----------
        n_nodes : integer
            The number of desired integration nodes.
        interval : numpy.ndarray(size=2)|None
            Interval of desired integration nodes.
            If unset (i.e. ``None``), default nodes interval is implementation
            dependent.

        Notes
        -----
        The implementation and behaviour must and will be defined by
        specializations of this interface.

        See Also
        --------
        .interval
            Accessor for the interval.
        """
        pass

    def transform(self):
        """
        Summary
        -------
        Transforms computed integration nodes to fit stored interval.
        """
        # TODO: implement transformation of integration nodes

    @property
    def interval(self):
        """
        Summary
        -------
        Accessor for the interval of the integration nodes.

        Parameters
        ----------
        interval : numpy.ndarray(size=2)
            Desired interval of integration nodes.

        Raises
        ------
        ValueError
            If ``interval`` is not an ``numpy.ndarray`` and not of size 2.

        Returns
        -------
        node interval : numpy.ndarray(size=2)
        """
        return self._interval

    @interval.setter
    def interval(self, interval):
        if not isinstance(interval, np.ndarray) or interval.size != 2:
            ValueError(self.__class__ + ".interval(): " +
                       "Given interval is not a numpy.ndarray or "
                       "is not of size 2: {:s} ({:s})"
                       .format(interval, type(interval)))
        self._interval = interval

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
