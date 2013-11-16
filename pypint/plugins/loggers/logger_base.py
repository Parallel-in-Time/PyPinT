# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class LoggerBase(object):
    def __init__(self):
        self._sink = None
        self._level = 5
        self._format = None

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
        return self._sink

    @sink.setter
    def sink(self, sink):
        self._sink = sink

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level):
        self._level = level

    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, format):
        self._format = format
