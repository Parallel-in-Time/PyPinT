# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from copy import deepcopy

import numpy as np

from pypint.utilities import class_name


class IWeightFunction(object):
    """Provider for integration weights functions.

    This is an abstract interface for providers of integration weights functions.
    """
    def __init__(self):
        self._weights = None
        self._interval = None

    def init(self, *args, **kwargs):
        """Sets and defines the weights function.

        Notes
        -----
        The implementation and behaviour must and will be defined by specializations of this interface.
        Implementations are allowed to add further named arguments.
        """
        pass

    def evaluate(self, nodes, interval=None):
        """Computes weights for given nodes based on set weight function.

        Parameters
        ----------
        nodes : :py:class:`numpy.ndarray`
            Array of nodes to compute weights for.

        interval : :py:class:`numpy.ndarray` or :py:class:`None`
            Array with the interval boundaries.
            If :py:class:`None` the boundaries of the given nodes are used.

        Returns
        -------
        computed weights : :py:class:`numpy.ndarray`
            Vector of computed weights.

        Notes
        -----
        The implementation and behaviour must and will be defined by specializations of this interface.
        """
        if interval is None:
            self._interval = np.array([nodes[0], nodes[-1]])
        else:
            self._interval = interval

    @property
    def weights(self):
        """Accessor for cached computed weights.

        Returns
        -------
        computed weights : :py:class:`numpy.ndarray`
            Cached computed weights.
        """
        return self._weights

    def print_lines_for_log(self):
        _lines = {
            'Type': class_name(self)
        }
        return _lines

    def __str__(self):
        return "IWeightFunction<0x%x>(weights=%s)" % (id(self), self.weights)

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
