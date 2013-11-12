# coding=utf-8
"""
An implementation of the *Spectral Deferred Corrections* (SDC) algorithm for
solving ordinary differential equations (ODE) in the Python programming
language.

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .integrate import *
from .sdc import SDC
from .globals import *

__all__ = ["sdc"]
