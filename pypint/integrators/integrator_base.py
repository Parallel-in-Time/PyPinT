# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np

from pypint.integrators.node_providers.i_nodes import INodes
from pypint.integrators.weight_function_providers.i_weight_function import IWeightFunction
from pypint.utilities import assert_is_instance, assert_condition, class_name


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
            Type of integration nodes as the class name **not instance**.
        num_nodes : :py:class:`int`
            Number of integration nodes
        weights_function : :py:class:`.IWeightFunction` or :py:class:`dict`
            Weight function for the integration nodes.
            If it is a dictionary, it must have a ``class`` field with the :py:class:`.IWeightFunction` as the value.
            Further fields are used as parameters to :py:class:`.IWeightFunction.init`.
            In both cases the weight function class must be given as a name **not instance**.

        Raises
        ------
        ValueError :
            If the type of one of the given arguments does not match.

            * ``nodes_type`` must be a subclass of :py:class:`.INodes`
            * ``num_nodes`` must be an :py:class:`int`
            * ``weights_function`` must be a subclass of :py:class:`.IWeightFunction` or :py:class:`dict`

                - if ``weights_function`` is a dictionary, its field ``class`` must be an :py:class:`.IWeightFunction`.

        Examples
        --------
        >>> from pypint.integrators import INTEGRATOR_PRESETS
        >>> integrator = IntegratorBase()
        >>> # create classic Gauss-Lobatto integrator with 4 integration nodes
        >>> options = INTEGRATOR_PRESETS['Gauss-Lobatto']
        >>> options['num_nodes'] = 4
        >>> integrator.init(**options)
        """
        assert_condition(isinstance(nodes_type, type) and issubclass(nodes_type, INodes),
                         ValueError, message="Given nodes type is not a valid type: %s"
                                             % nodes_type.__mro__[-2].__name__,
                         checking_obj=self)
        if isinstance(weights_function, dict):
            assert_condition('class' in weights_function and issubclass(weights_function['class'], IWeightFunction),
                             ValueError, "Given weight function is not a valid type: %s"
                                         % weights_function['class'].__mro__[-2].__name__,
                             self)
            self._weights_function = weights_function['class']()
            # copy() is necessary as dictionaries are passed by reference
            _weight_function_options = weights_function.copy()
            del _weight_function_options['class']
            self._weights_function.init(**_weight_function_options)
        else:
            assert_condition(issubclass(weights_function, IWeightFunction),
                             ValueError, message="Given Weight Function is not a valid type: %s"
                                                 % weights_function.__mro__[-2].__name__,
                             checking_obj=self)
            self._weights_function = weights_function()
            self._weights_function.init()
        assert_is_instance(num_nodes, int,
                           "Number of nodes need to be an integer: NOT %s" % type(num_nodes),
                           self)
        self._nodes = nodes_type()
        self._nodes.init(num_nodes, interval=interval)
        self._weights_function.evaluate(self._nodes.nodes, interval=self._nodes.interval)

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
        assert_is_instance(data, np.ndarray, descriptor="Data to integrate", checking_obj=self)
        assert_condition("time_start" in kwargs or "time_end" in kwargs,
                         ValueError, message="Either start or end of time interval need to be given.",
                         checking_obj=self)
        assert_condition(kwargs["time_start"] < kwargs["time_end"],
                         ValueError,
                         message="Time interval need to be non-zero positive: [{:f}, {:f}]"
                                 .format(kwargs["time_start"], kwargs["time_end"]),
                         checking_obj=self)

    def transform_interval(self, interval):
        """Transform current interval to the given one

        The integration nodes are transformed to fit the given interval and subsequently the weights are recomputed to
        match the new nodes.

        See Also
        --------
        :py:meth:`.INodes.transform` : method called for node transformation
        """
        if interval is not None:
            self._nodes.interval = interval
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
        print("%s<0x%x>: %s" % (type(self._weights_function), id(self._weights_function), self._weights_function))
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
    def num_nodes(self):
        return self._nodes.num_nodes

    @property
    def weights_function(self):
        """Read-only accessor for the weights function

        Returns
        -------
        weights_function : :py:class:`.IWeightFunction`
        """
        return self._weights_function

    def print_lines_for_log(self):
        _lines = {
            'Type': class_name(self)
        }
        if self._nodes is not None:
            _lines['Nodes'] = self._nodes.print_lines_for_log()
        if self._weights_function is not None:
            _lines['Weight Function'] = self._weights_function.print_lines_for_log()
        return _lines

    def __str__(self):
        return "IntegratorBase<0x%x>(nodes=%s, weights=%s)" % (id(self), self.nodes_type, self.weights_function)
