"""
Integration Routines
"""

from .quadrature import Quadrature
from .gauss import Gauss
from .newton_cotes import NewtonCotes

__all__ = ["quadrature", "newton_cotes", "gauss"]
