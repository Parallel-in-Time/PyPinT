# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np

from pypint.integrators.node_providers.i_nodes import INodes
from pypint.integrators.weight_function_providers.i_weight_function import IWeightFunction
from pypint.utilities import assert_is_instance, assert_condition


class IntegratorBase(object):
    """Basic and generic integrator for variable nodes and weights.
    """
    def __init__(self):
        self._nodes = None
        self._weights_function = None

    def init(self, nodes_type, num_nodes, weights_function, interval=None):
        """Initializes the integrator with given nodes and weights function.

        Before setting the given attributes, a consistency check is carried out testing for the correct types.
        :py:meth:`.INodes.init` is called on the provided nodes object.
        :py:meth:`.IWeightFunction.evaluate` is called on the provided weight function object.

        Parameters
        ----------
        nodes_type : :py:class:`.INodes`
            Type of integration nodes.

        num_nodes : :py:class:`int`
            Number of integration nodes

        weights_function : :py:class:`.IWeightFunction` or :py:class:`dict`
            Weight function for the integration nodes.
            If it is a dictionary, it must have a ``class`` field with the :py:class:`.IWeightFunction` as the value.
            Further fields are used as parameters to :py:class:`.IWeightFunction.init`

        Raises
        ------
        ValueError :
            If the type of one of the given arguments does not match.

            * ``nodes_type`` must be an :py:class:`.INodes`
            * ``num_nodes`` must be an :py:class:`int`
            * ``weights_function`` must be an :py:class:`.IWeightFunction` or :py:class:`dict`
                - If ``weights_function`` is a dictionary, its field ``class`` must be an :py:class:`.IWeightFunction`.

        Examples
        --------
        >>> from pypint.integrators import INTEGRATOR_PRESETS
        >>> integrator = IntegratorBase()
        >>> # create classic Gauss-Lobatto integrator with 4 integration nodes
        >>> options = INTEGRATOR_PRESETS["Gauss-Lobatto"]
        >>> options["num_nodes"] = 4
        >>> integrator.init(**options)
        """

        assert_is_instance(nodes_type, INodes,
                           "Given nodes type is not a valid type: %s" % type(nodes_type),
                           self)
        if isinstance(weights_function, dict):
            assert_condition("class" in weights_function or isinstance(weights_function["class"], IWeightFunction),
                             ValueError, "Given weight function is not a valid type: %s" % type(weights_function),
                             self)
            self._weights_function = weights_function["class"]
            # copy() is necessary as dictionaries are passed by reference
            _weight_function_options = weights_function.copy()
            del _weight_function_options["class"]
            self._weights_function.init(**_weight_function_options)
        else:
            assert_is_instance(weights_function, IWeightFunction,
                               "Given weight function is not a valid type: %s" % type(weights_function),
                               self)
            self._weights_function = weights_function
            self._weights_function.init()
        assert_is_instance(num_nodes, int,
                           "Number of nodes need to be an integer: NOT %s" % type(num_nodes),
                           self)
        self._nodes = nodes_type
        self._nodes.init(num_nodes)
        self.transform_interval(interval)

    def evaluate(self, data, **kwargs):
        """Applies this integrator to given data in specified time interval.

        Parameters
        ----------
        data : :py:class:`numpy.ndarray`
            Data vector of the values at given time points.
            Its length must equal the number of integration nodes.

        time_start : :py:class:`float`
            *(optional)*
            Begining of the time interval to integrate over.

        time_end : :py:class:`float`
            *(optional)*
            End of the time interval to integrate over.

        Raises
        ------
        ValueError :

            * if ``data`` is not a :py:class:`numpy.ndarray`
            * if either ``time_start`` or ``time_end`` are not given
            * if ``time_start`` is larger or equals ``time_end``
        """
        assert_is_instance(data, np.ndarray, "Data to integrate must be an numpy.ndarray.", self)
        assert_condition("time_start" in kwargs or "time_end" in kwargs,
                         ValueError, "Either start or end of time interval need to be given.", self)
        assert_condition(kwargs["time_start"] < kwargs["time_end"],
                         ValueError, "Time interval need to be non-zero positive: [{:f}, {:f}]"
                                     .format(kwargs["time_start"], kwargs["time_end"]),
                         self)

    def transform_interval(self, interval):
        """Transform current interval to the given one

        The integration nodes are transformed to fit the given interval and subsequently the weights are recomputed to
        match the new nodes.

        See Also
        --------
        :py:meth:`.INodes.transform` : method called for node transformation
        """
        if interval is not None:
            self._nodes.transform(interval)
        self._weights_function.evaluate(self._nodes.nodes)

    @property
    def nodes(self):
        """Proxy accessor for the integration nodes.

        See Also
        --------
        :py:attr:`.INodes.nodes`
        """
        return self._nodes.nodes

    @property
    def weights(self):
        """Proxy accessor for the calculated and cached integration weights.

        See Also
        --------
        :py:attr:`.IWeightFunction.weights`
        """
        return self._weights_function.weights

    @property
    def nodes_type(self):
        """Read-only accessor for the type of nodes

        Returns
        -------
        nodes : :py:class:`.INodes`
        """
        return self._nodes

    @property
    def weights_function(self):
        """Read-only accessor for the weights function

        Returns
        -------
        weights_function : :py:class:`.IWeightFunction`
        """
        return self._weights_function

    def print_lines_for_log(self):
        _lines = {}
        _lines['Type'] = self.__class__.__name__
        if self._nodes is not None:
            _lines['Nodes'] = self._nodes.print_lines_for_log()
        if self._weights_function is not None:
            _lines['Weight Function'] = self._weights_function.print_lines_for_log()
        return _lines
