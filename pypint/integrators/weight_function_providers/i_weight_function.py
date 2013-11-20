# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class IWeightFunction(object):
    """
    Summary
    -------
    Provider for integration weights functions.

    Extended Summary
    ----------------
    This is an abstract interface for providers of integration weights
    functions.
    """
    def __init__(self):
        self._weights = None

    def init(self, *args, **kwargs):
        """
        Summary
        -------
        Sets and defines the weights function.

        Parameters
        ----------
        args, kwargs
            Implementation defined type to specify the weight function's
            parameters.

        Notes
        -----
        The implementation and behaviour must and will be defined by
        specializations of this interface.
        Implementations are allowed to add further named arguments.
        """
        pass

    def evaluate(self, nodes, interval = None):
        """
        Summary
        -------
        Computes weights for given nodes based on set weight function.

        Extended Summary
        ----------------

        Parameters
        ----------
        nodes : numpy.ndarray
            Array of nodes to compute weights for.
        interval : numpy.ndarray
            Array with the interval boundaries. If None one uses the the boundaries in nodes
        Returns
        -------
        computed weights : numpy.ndarray
            Vector of computed weights.

        Notes
        -----
        The implementation and behaviour must and will be defined by
        specializations of this interface.
        """
        pass

    @property
    def weights(self):
        """
        Summary
        -------
        Accessor for cached computed weights,.

        Returns
        -------
        computed weights : numpy.ndarray
            Cached computed weights.
        Notes
        -----
        The weights are calculated with help of the Lagrangepolynomials
        .. math::
        \alpha_i = \int_{a)^{b}\omega (x) \Pi_{j=1,j \neq i}^{n} \frac{x - x_j}{x_i - x_j} \mathrm{d}x

        """
        return self._weights
