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


LOG = Logger('PyPinT Logging')
LOG.handlers = [
    # log ERRORS and WARNINGS to stderr
    ColorizedStderrHandler(level='WARNING',
                           format_string='[{record.level_name: <8s}] {record.module:s}.{record.func_name:s}(): '
                                         '{record.message:s}',
                           bubble=True),
    # then write all ERROR, WARNING and INFO messages to stdout
    StreamHandler(stdout, level='INFO',
                  format_string='[{record.level_name: <8s}] {record.message:s}',
                  bubble=True),
    # finally, write everything (including DEBUG messages) to a logfile
    FileHandler('{:%Y-%m-%d_%H-%M-%S}_debug.log'.format(datetime.now()), level='DEBUG',
                format_string='[{record.time}] [{record.level_name: <8s}] <{record.process}.{record.thread}> '
                              '{record.module:s}.{record.func_name:s}():{record.lineno:d}: {record.message:s}')
]


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
        LOG.debug("%s<0x%x>.%s(%s): " % (checking_obj_name(obj), id(obj), inspect.stack()[2][3], _params) + add_log_msg)
    else:
        LOG.debug("unknown<>.%s(%s): " % (inspect.stack()[2][3], _params) + add_log_msg)


__all__ = [
    'LOG',
    'VERBOSITY_LVL1', 'VERBOSITY_LVL2', 'VERBOSITY_LVL3',
    'SEPARATOR_LVL1', 'SEPARATOR_LVL2', 'SEPARATOR_LVL3',
    'this_got_called', 'print_logging_message_tree'
]
