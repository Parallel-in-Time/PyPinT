# coding=utf-8
"""
*PyPinT* is a framework for Parallel-in-Time integration routines.

The main purpose of *PyPinT* is to provide a framework for educational use and
prototyping new parallel-in-time algorithms.
As well it will aid in developing a high-performance C++ implementation for
massively parallel computers providing the benefits of parallel-in-time
routines to a zoo of time integrators in various applications.

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
.. moduleauthor:: Dieter Moser <d.moser@fz-juelich.de>
"""

from .problems import *
from .solutions import *
from .integrators import *
from .multi_level_providers import *
from .solvers import *
from .communicators import *
from .plugins import *

__all__ = ['problems', 'solutions', 'integrators', 'multi_level_providers',
           'solvers', 'communicators', 'plugins']
