# coding=utf-8
"""Logging Framework for PyPinT

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import logging as log
from sys import stdout
from collections import OrderedDict

loggers = []

VERBOSITY_LVL1 = '!> '
VERBOSITY_LVL2 = '>> '
VERBOSITY_LVL3 = ' > '

SEPARATOR_LVL1 = '#' * 80
SEPARATOR_LVL2 = '-' * 80
SEPARATOR_LVL3 = '.' * 80


def print_logging_message_tree(messages):
    for _key1, _value1 in messages.items():
        if isinstance(_value1, (dict, OrderedDict)):
            if _key1 != '':
                logger().info("%s%s:" % (VERBOSITY_LVL1, _key1))
            for _key2, _value2 in _value1.items():
                if isinstance(_value2, (dict, OrderedDict)):
                    if _key2 != '':
                        logger().info("%s  %s:" % (VERBOSITY_LVL1, _key2))
                    for _key3, _value3 in _value2.items():
                        if isinstance(_value3, (dict, OrderedDict)):
                            if _key3 != '':
                                logger().info("%s    %s:" % (VERBOSITY_LVL1, _key3))
                            for _key4, _value4 in _value3.items():
                                logger().info("{}      {: <28s} {}".format(VERBOSITY_LVL1, "%s: " % _key4, _value4))
                        else:
                            logger().info("{}    {: <30s} {}".format(VERBOSITY_LVL1, "%s: " % _key3, _value3))
                else:
                    logger().info("{}  {: <32s} {}".format(VERBOSITY_LVL1, "%s: " % _key2, _value2))
        else:
            logger().info("{}{: <34s} {}".format(VERBOSITY_LVL1, "%s: " % _key1, _value1))


def logger():
    """Accessor for a standard logger
    """
    global loggers

    if len(loggers) > 0:
        return loggers[0]

    else:
        _logger = log.getLogger("ConsoleLogger")
        _logger.setLevel(log.DEBUG)
        _handler = log.StreamHandler(stdout)
        _formatter = log.Formatter('%(levelname)s %(module)s.%(funcName)s: %(message)s',
                                   '%d.%m.%y %H:%M:%S %Z')
        _handler.setFormatter(_formatter)
        _logger.addHandler(_handler)
        loggers.append(_logger)
        return logger()


__all__ = [
    'VERBOSITY_LVL1', 'VERBOSITY_LVL2', 'VERBOSITY_LVL3',
    'SEPARATOR_LVL1', 'SEPARATOR_LVL2', 'SEPARATOR_LVL3',
    'print_logging_message_tree'
]
