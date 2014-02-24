# coding=utf-8
"""*PyPinT* is a framework for Parallel-in-Time integration routines.

The main purpose of *PyPinT* is to provide a framework for educational use and prototyping new parallel-in-time
algorithms.
As well it will aid in developing a high-performance C++ implementation for massively parallel computers providing the
benefits of parallel-in-time routines to a zoo of time integrators in various applications.

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
.. moduleauthor:: Dieter Moser <d.moser@fz-juelich.de>
"""

from sys import version_info
if version_info.major < 3:
    raise RuntimeError("PyPinT requires Python 3.x")


__version__ = '0.0.1'


# initialize Logging framework
from pypint.utilities.logging import logger
LOG = logger()
"""Easy accessor for the currently configured logger.

See Also
--------
:py:meth:`.utilities.logging.Logging.logger` : aliased accessor function
"""

__all__ = ["LOG"]
