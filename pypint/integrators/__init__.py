# coding=utf-8
"""Integrators for Iterative Time Solvers

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
.. moduleauthor:: Dieter Moser <d.moser@fz-juelich.de>
"""
from pypint.integrators.node_providers.gauss_lobatto_nodes import GaussLobattoNodes
from pypint.integrators.node_providers.gauss_legendre_nodes import GaussLegendreNodes
from pypint.integrators.weight_function_providers.polynomial_weight_function import PolynomialWeightFunction


INTEGRATOR_PRESETS = {}
"""Useful presets for integrators.

This dictionary provides useful parameter presets for frequently used integrators.
Use them as parameters to the constructor of :py:class:`.IntegratorBase`.
Available presets:

**Gauss-Lobatto**
    Classic *Gauss-Lobatto* integrator with constant one polynomial as the weight function.

Examples
--------
>>> integrator_base_params = INTEGRATOR_PRESETS["Gauss-Lobatto"]
"""


INTEGRATOR_PRESETS["Gauss-Lobatto"] = {
    "nodes_type": GaussLobattoNodes(),
    "weights_function": {
        "class": PolynomialWeightFunction(),
        "coeffs": [1.0]
    },
    "num_nodes": 3
}


INTEGRATOR_PRESETS["Gauss-Legendre"] = {
    "nodes_type": GaussLegendreNodes(),
    "weights_function": {
        "class": PolynomialWeightFunction(),
        "coeffs": [1.0]
    },
    "num_nodes": 3
}
