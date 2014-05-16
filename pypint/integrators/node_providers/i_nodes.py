# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from copy import deepcopy

import numpy as np

from pypint.utilities import assert_is_instance, assert_condition, class_name
from pypint.utilities.logging import LOG


class INodes(object):
    """Provider for integration nodes.

    This is an abstract interface for providers of integration nodes.
    """

    std_interval = np.array([0.0, 0.0])
    """Standard interval for this integration nodes.
    """

    def __init__(self):
        self._num_nodes = None
        self._nodes = None
        self._interval = None

    def init(self, n_nodes, interval=None):
        """Initializes the vector of integration nodes of size `n_nodes`.

        Parameters
        ----------
        n_nodes : :py:class:`int`
            The number of desired integration nodes.

        interval : :py:class:`numpy.ndarray(size=2)` or :py:class:`None`
            Interval of desired integration nodes.
            If unset (i.e. :py:class:`None`), default nodes interval is implementation dependent.

        Notes
        -----
        The implementation and behaviour must and will be defined by specializations of this interface.

        See Also
        --------
        :py:attr:`.interval` : Accessor for the interval.
        """
        pass

    def transform(self, interval):
        """Transforms computed integration nodes to fit a new given interval.

        Based on the old interval the computed integration nodes are transformed fitting the newly given interval using
        standard linear interval scaling.
        In case no interval was previously given, the standard interval of the used nodes method, e.g. :math:`[-1, 1]`
        for Gauss-Lobatto, is used.

        Parameters
        ----------
        interval : :py:class:`numpy.ndarray(size=2)`
            New interval to transform nodes onto.

        Raises
        ------
        ValueError
            If the standard interval is not suited for transformation, i.e. it is not a :py:class:`numpy.ndarray` of
            size 2 and not positive.

        Notes
        -----
        It may be this transformation is numerically inconvenient because of the loss of significance.
        """
        assert_is_instance(interval, np.ndarray, descriptor="Interval", checking_obj=self)
        assert_condition(interval.size == 2,
                         ValueError,
                         message="Intervals must be of size 2: {} ({:s})".format(interval, class_name(interval)),
                         checking_obj=self)
        assert_condition(interval[0] < interval[1],
                         ValueError,
                         message="Interval must be positive: {:.2f} > {:.2f}".format(interval[0], interval[1]),
                         checking_obj=self)
        _old_interval = self.interval
        self._interval = interval
        self._nodes = (self.nodes - _old_interval[0]) * (interval[1] - interval[0]) / \
                      (_old_interval[1] - _old_interval[0]) + interval[0]
        assert_condition(self._nodes[0] - self._interval[0] <= 1e-16 and self._nodes[-1] - self._interval[1] <= 1e-16,
                         RuntimeError,
                         message="Newly computed nodes do not match new interval: {} NOT IN {}"
                                 .format(self._nodes, self._interval),
                         checking_obj=self)
        LOG.debug("Nodes: %s" % self._nodes.tolist())

    @property
    def interval(self):
        """Accessor for the interval of the integration nodes.

        Parameters
        ----------
        interval : :py:class:`numpy.ndarray(size=2)`
            Desired interval of integration nodes.

        Raises
        ------
        ValueError :
            If ``interval`` is not an :py:class:`numpy.ndarray` and not of size 2.

        Returns
        -------
        node_interval : :py:class:`numpy.ndarray(size=2)`
            Interval of the nodes.

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
        """Accessor for the vector of integration nodes.

        Returns
        -------
        nodes : :py:class:`numpy.ndarray`
            Vector of nodes.
        """
        return self._nodes

    @property
    def num_nodes(self):
        """Accessor for the number of desired integration nodes.

        Returns
        -------
        number of nodes : :py:class:`int`
            The number of desired and/or computed integration nodes.

        Notes
        -----
        Specializations of this interface might override this accessor.
        """
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self._num_nodes = num_nodes

    def print_lines_for_log(self):
        _lines = {
            'Type': class_name(self),
            'Number Nodes': "%d" % self.num_nodes
        }
        return _lines

    def __str__(self):
        return "INodes<0x%x>(n=%d, nodes=%s)" % (id(self), self.num_nodes, self.nodes)

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
