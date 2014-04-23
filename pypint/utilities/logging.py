# coding=utf-8
"""Logging Framework for PyPinT

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from logbook import Logger, StreamHandler, FileHandler
from logbook.more import ColorizedStderrHandler
from sys import stdout
from datetime import datetime
from collections import OrderedDict
import inspect

from pypint.utilities.tracing import checking_obj_name
from pypint.utilities.config import config


LOG = Logger('PyPinT Logging')
LOG.handlers = []


if config()['Logger']['Stderr']['enable']:
    # log ERRORS and WARNINGS to stderr
    LOG.handlers.append(
        ColorizedStderrHandler(level=config()['Logger']['Stderr']['level'],
                               format_string=config()['Logger']['Stderr']['format_string'],
                               bubble=config()['Logger']['Stderr']['bubble'])
    )
if config()['Logger']['Stdout']['enable']:
    # then write all ERROR, WARNING and INFO messages to stdout
    LOG.handlers.append(
        StreamHandler(stdout,
                      level=config()['Logger']['Stdout']['level'],
                      format_string=config()['Logger']['Stdout']['format_string'],
                      bubble=config()['Logger']['Stdout']['bubble'])
    )
if config()['Logger']['File']['enable']:
    # finally, write everything (including DEBUG messages) to a logfile
    LOG.handlers.append(
        FileHandler(config()['Logger']['File']['file_name_format'].format(datetime.now()),
                    level=config()['Logger']['File']['level'],
                    format_string=config()['Logger']['File']['format_string'])
    )


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
                LOG.info("%s%s:" % (VERBOSITY_LVL1, _key1))
            for _key2, _value2 in _value1.items():
                if isinstance(_value2, (dict, OrderedDict)):
                    if _key2 != '':
                        LOG.info("%s  %s:" % (VERBOSITY_LVL1, _key2))
                    for _key3, _value3 in _value2.items():
                        if isinstance(_value3, (dict, OrderedDict)):
                            if _key3 != '':
                                LOG.info("%s    %s:" % (VERBOSITY_LVL1, _key3))
                            for _key4, _value4 in _value3.items():
                                LOG.info("{}      {: <28s} {}".format(VERBOSITY_LVL1, "%s: " % _key4, _value4))
                        else:
                            LOG.info("{}    {: <30s} {}".format(VERBOSITY_LVL1, "%s: " % _key3, _value3))
                else:
                    LOG.info("{}  {: <32s} {}".format(VERBOSITY_LVL1, "%s: " % _key2, _value2))
        else:
            LOG.info("{}{: <34s} {}".format(VERBOSITY_LVL1, "%s: " % _key1, _value1))


def this_got_called(obj, *args, add_log_msg="", **kwargs):
    _params = ''
    if len(args) > 0:
        _params += ', '.join(args)
        if len(kwargs) > 0:
            _params += ', '
    if len(kwargs) > 0:
        _c = 0
        for _k in kwargs:
            if _c > 0:
                _params += ', '
            _params += str(_k) + '=' + str(kwargs[_k])
            _c += 1

    if obj:
        LOG.debug("%s<0x%x>.%s(%s): " % (checking_obj_name(obj), id(obj), inspect.stack()[1][3], _params) + add_log_msg)
    else:
        LOG.debug("unknown<>.%s(%s): " % (inspect.stack()[1][3], _params) + add_log_msg)


__all__ = [
    'LOG',
    'VERBOSITY_LVL1', 'VERBOSITY_LVL2', 'VERBOSITY_LVL3',
    'SEPARATOR_LVL1', 'SEPARATOR_LVL2', 'SEPARATOR_LVL3',
    'this_got_called', 'print_logging_message_tree'
]
