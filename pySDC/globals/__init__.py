__all__ = ["LOG", "DIGITS", "PRECISION"]

import logging
import logging.config
from yaml import load

logging.config.dictConfig(load(open("logging_config.yaml", 'r')))
LOG = logging.getLogger("consoleLogger")
"""
"""

DIGITS = 12
"""
"""

PRECISION = 1e-7
"""
"""
