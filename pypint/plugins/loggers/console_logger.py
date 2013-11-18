# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.plugins.loggers.logger_base import LoggerBase

import sys as sys
import logging as logging


class ConsoleLogger(LoggerBase):
    """
    A basic console logger.

    It wrapps Python's buildin logging library with the default logging format::

        levelname module.funcName: message
    """
    def __init__(self):
        super(self.__class__, self).__init__()
        logging.basicConfig(stream=sys.stdout)
        self._sink = logging.getLogger('ConsoleLogger')
        self.level = logging.INFO
        self.format = '%(levelname)s %(module)s.%(funcName)s: %(message)s'

    def info(self, message):
        """
        Loggs the given message on the ``INFO`` level.

        :param message: message to log
        :type message:  string
        """
        self._sink.info(message)

    def debug(self, message):
        """
        Loggs the given message on the ``DEBUG`` level.

        :param message: debugging message to log
        :type message:  string
        """
        self._sink.debug(message)

    def warn(self, message):
        """
        Loggs the given message on the ``WARNING`` level.

        :param message: warning message to log
        :type message:  string
        """
        self._sink.warning(message)

    def error(self, message):
        """
        Loggs the given message on the ``ERROR`` level.

        :param message: error message to log
        :type message:  string
        """
        self._sink.error(message)

    def critical(self, message):
        """
        Loggs the given message on the ``CRITICAL`` level.

        :param message: critical message to log
        :type message:  string
        """
        self._sink.critical(message)

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level):
        self._level = level
        self._sink.setLevel(level)

    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, format):
        self._format = format
        logging.basicConfig(format=self.format)
