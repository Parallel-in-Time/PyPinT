# coding=utf-8
"""
Integrators for Iterative Time Solvers

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
.. moduleauthor:: Dieter Moser <d.moser@fz-juelich.de>
"""

from .node_providers.gauss_lobatto_nodes import GaussLobattoNodes
from .node_providers.gauss_legendre_nodes import GaussLegendreNodes
from .weight_function_providers.polynomial_weight_function import PolynomialWeightFunction

#: Summary
#: -------
#: Useful presets for integrators.
#:
#: Extended Summary
#: ----------------
#: This dictionary provides useful parameter presets for frequently used
#: integrators.
#: Use them as parameters to the constructor of :py:class:`.IntegratorBase`.
#: Available presets:
#:
#: **Gauss-Lobatto**
#:     Classic *Gauss-Lobatto* integrator with constant one polynomial as the
#:      weight function.
#:
#: Examples
#: --------
#: >>> integrator_base_params = integrator_presets["Gauss-Lobatto"]
integrator_presets = {
    "Gauss-Lobatto": {
        "nodes_type": GaussLobattoNodes(),
        "weights_function": {
            "class": PolynomialWeightFunction(),
            "coeffs": [1.0]
        },
        "num_nodes": 3
    },
    "Gauss-Legendre": {
        "nodes_type": GaussLegendreNodes(),
        "weights_function": {
            "class": PolynomialWeightFunction(),
            "coeffs": [1.0]
        },
        "num_nodes": 3
    }
}
