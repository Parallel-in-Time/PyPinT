"""
General global variables for pySDC such as the logging backend
"""
import logging
import logging.config
import os
from yaml import load

logging.config.dictConfig(load(open(os.path.dirname(os.path.abspath(__file__))
                                    + "/../../logging_config.yaml", 'r')))
LOG = logging.getLogger("consoleLogger")
"""
"""

DIGITS = 12
"""
"""

PRECISION = 1e-7
"""
"""

__all__ = ["LOG", "DIGITS", "PRECISION"]
