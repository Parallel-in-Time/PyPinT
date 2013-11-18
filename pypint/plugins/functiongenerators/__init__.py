"""
Function generators
"""

__author__ = 'moser'

from .pypint.plugins.functiongenerators.piecewise import PiecewiseFG
from .pypint.plugins.functiongenerators.polynomial import PolynomialFG
from .pypint.plugins.functiongenerators.trigonometric import TrigonometricFG

__all__=["piecewise","polynomial","trigonometric","nested"]
