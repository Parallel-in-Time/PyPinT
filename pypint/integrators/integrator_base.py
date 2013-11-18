# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .node_providers.i_nodes import INodes
from .weight_function_providers.i_weight_function import IWeightFunction


class IntegratorBase(object):
    """
    Summary
    -------
    Basic and generic integrator for variable nodes and weights.

    Extended Summary
    ----------------
    """
    def __init__(self):
        self._nodes = None
        self._weights_function = None

    def init(self, nodes_type, num_nodes, weights_function):
        """
        Summary
        -------
        Initializes the integrator with given nodes and weights function.

        Extended Summary
        ----------------
        Before setting the given attributes, a consistency check is carried out
        testing for the correct types.
        :py:meth:`.INodes.init` is called on the provided nodes object.
        :py:meth:`.IWeightFunction.evaluate` is called on the provided weight
        function object.

        Parameters
        ----------
        nodes_type : INodes
            Type of integration nodes.
        num_nodes : integer
            Number of integration nodes
        weights_function : IWeightFunction|dict
            Weight function for the integration nodes.
            If it is a dictionary, it must have a ``class`` field with the
            :py:class:`.IWeightFunction` as the value.
            Further fields are used as parameters to
            :py:class:`.IWeightFunction.init`

        Raises
        ------
        ValueError
            If the type of one of the given arguments does not match.

            * ``nodes_type`` must be an :py:class:`.INodes`
            * ``num_nodes`` must be an integer
            * ``weights_function`` must be an :py:class:`.IWeightFunction` or
              dictionary
              * If ``weights_function`` is a dictionary, its field ``class``
                must be an :py:class:`.IWeightFunction`.

        Examples
        --------
        >>> from pypint.integrators import integrator_presets
        >>> integrator = IntegratorBase()
        >>> # create classic Gauss-Lobatto integrator with four integration nodes
        >>> options = integrator_presets["Gauss-Lobatto"]
        >>> options["num_nodes"] = 4
        >>> integrator.init(**options)
        """
        if not isinstance(nodes_type, INodes):
            raise ValueError(self.__class__ + ".init(): " +
                             "Given nodes type is not a valid type: {}"
                             .format(nodes_type.__name__))
        if isinstance(weights_function, dict):
            if not isinstance(weights_function["class"], IWeightFunction):
                raise ValueError(self.__class__ + ".init(): " +
                                 "Given weight function is not a valid type: {}"
                                 .format(weights_function.__name__))
        if not isinstance(num_nodes, int):
            raise ValueError(self.__class__ + ".init(): " +
                             "Number of nodes need to be an integer (not {})."
                             .format(num_nodes.__name__))
        self._nodes = nodes_type
        self._nodes.init(num_nodes)
        self._weights_function = weights_function["class"]
        _weight_function_options = weights_function
        del _weight_function_options["class"]
        self._weights_function.init(**_weight_function_options)
        self._weights_function.evaluate(self._nodes.nodes)

    def evaluate(self, data, time_start, time_end):
        """
        Summary
        -------
        Applies this integrator to given data in specified time interval.

        Extended Summary
        ----------------

        Parameters
        ----------
        data : numpy.ndarray|function pointer
            Data vector or pointer to a function returning the values at given
            time points.
            If a vector is given, its length must equal the number of
            integration nodes.
        time_start : float
            Begining of the time interval to integrate over.
        time_end : float
            End of the time interval to integrate over.
        """
        pass

    @property
    def nodes(self):
        """
        Summary
        -------
        Proxy accessor for the integration nodes.

        See Also
        --------
        .INodes.nodes
        """
        return self._nodes.nodes

    @property
    def weights(self):
        """
        Summary
        -------
        Proxy accessor for the calculated and cached integration weights.

        See Also
        --------
        .IWeightFunction.weights
        """
        return self._weights_function.weights
