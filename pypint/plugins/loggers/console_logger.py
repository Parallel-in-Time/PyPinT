# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .logger_base import *

import sys as sys
import logging as logging


class ConsoleLogger(LoggerBase):
    """
    A basic console logger.

    It wrapps Python's buildin logging library with the default logging format::

        levelname module.funcName: message
    """
    def __init__(self):
        super().__init__()
        logging.basicConfig(stream=sys.stdout)
        self.__sink = logging.getLogger('ConsoleLogger')
        self.level = logging.INFO
        self.format = '%(levelname)s %(module)s.%(funcName)s: %(message)s'

    def info(self, message):
        """
        Loggs the given message on the ``INFO`` level.

        :param message: message to log
        :type message:  string
        """
        self.__sink.info(message)

    def debug(self, message):
        """
        Loggs the given message on the ``DEBUG`` level.

        :param message: debugging message to log
        :type message:  string
        """
        self.__sink.debug(message)

    def warn(self, message):
        """
        Loggs the given message on the ``WARNING`` level.

        :param message: warning message to log
        :type message:  string
        """
        self.__sink.warning(message)

    def error(self, message):
        """
        Loggs the given message on the ``ERROR`` level.

        :param message: error message to log
        :type message:  string
        """
        self.__sink.error(message)

    def critical(self, message):
        """
        Loggs the given message on the ``CRITICAL`` level.

        :param message: critical message to log
        :type message:  string
        """
        self.__sink.critical(message)

    @property
    def level(self):
        return self.__level

    @level.setter
    def level(self, level):
        self.__level = level
        self.__sink.setLevel(level)

    @property
    def format(self):
        return self.__format

    @format.setter
    def format(self, format):
        self.__format = format
        logging.basicConfig(format=self.format)
