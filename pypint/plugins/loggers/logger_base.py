# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class LoggerBase(object):
    def __init__(self):
        self.__sink = None
        self.__level = 5
        self.__format = None

    def info(self, message):
        pass

    def debug(self, message):
        pass

    def warn(self, message):
        pass

    def error(self, message):
        pass

    def critical(self, message):
        pass

    @property
    def sink(self):
        return self.__sink

    @sink.setter
    def sink(self, sink):
        self.__sink = sink

    @property
    def level(self):
        return self.__level

    @level.setter
    def level(self, level):
        self.__level = level

    @property
    def format(self):
        return self.__format

    @format.setter
    def format(self, format):
        self.__format = format
