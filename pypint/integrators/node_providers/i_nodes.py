# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from copy import deepcopy
import numpy as np
from pypint.utilities import *
from pypint import LOG


class INodes(object):
    """
    Summary
    -------
    Provider for integration nodes.

    Extended Summary
    ----------------
    This is an abstract interface for providers of integration nodes.
    """

    std_interval = np.array([0.0, 0.0])
    """
    Summary
    -------
    Standard interval for this integration nodes.
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

    def transform(self, interval):
        """
        Summary
        -------
        Transforms computed integration nodes to fit a new given interval.

        Extended Summary
        ----------------
        Based on the old interval the computed integration nodes are transformed fitting the newly
        given interval using standard linear interval scaling.
        In case no interval was previously given, the standard interval of the used nodes method,
        e.g. :math:`[-1, 1]` for Gauss-Lobatto, is used.

        Parameters
        ----------
        interval : numpy.ndarray(size=2)
            New interval to transform nodes onto.

        Raises
        ------
        ValueError
            If the standard interval is not suited for transformation, i.e. it
            is not a ``numpy.ndarray`` of size 2 and not positive.

        Notes
        -----
        It may be this transformation is numerically inconvenient because of
        the loss of significance.
        """
        if not isinstance(interval, np.ndarray) or interval.size != 2:
            raise ValueError(func_name(self) +
                             "Given interval is not a numpy.ndarray or is not of size 2: {:s} ({:s})"
                             .format(interval, type(interval)))
        if interval[0] >= interval[1]:
            raise ValueError(func_name(self) +
                             "Given interval is not positive: {:.2f} > {:.2f}"
                             .format(interval[0], interval[1]))
        _old_interval = self.interval
        self._interval = interval
        self._nodes = (self.nodes - _old_interval[0]) * (interval[1] - interval[0]) / \
                      (_old_interval[1] - _old_interval[0]) + interval[0]
        #LOG.debug("Transformed nodes from {:s} -> {:s}: {:s}"
        #          .format(_old_interval, self.interval, self._nodes))

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

        Notes
        -----
        The setter calls :py:meth:`.transform` with the given interval.
        """
        return self._interval

    @interval.setter
    def interval(self, interval):
        self.transform(interval)

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
