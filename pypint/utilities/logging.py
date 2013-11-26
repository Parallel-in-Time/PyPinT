# coding=utf-8
"""
Logging Framework for PyPinT

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

import logging as log
from sys import stdout

_loggers = []


def logger():
    """
    Summary
    -------
    Accessor for a standard logger

    """
    global _loggers

    if len(_loggers) > 0:
        return _loggers[0]

    else:
        _logger = log.getLogger("ConsoleLogger")
        _logger.setLevel(log.DEBUG)
        _handler = log.StreamHandler(stdout)
        _formatter = log.Formatter('%(levelname)s %(module)s.%(funcName)s: %(message)s',
                                   '%d.%m.%y %H:%M:%S %Z')
        _handler.setFormatter(_formatter)
        _logger.addHandler(_handler)
        _loggers.append(_logger)
        return logger()
