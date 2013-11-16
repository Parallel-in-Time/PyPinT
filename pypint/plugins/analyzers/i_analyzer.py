# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.plugins.loggers.logger_base import LoggerBase


class IAnalyzer(object):
    def __init__(self):
        self._solutions = []
        self._logger = LoggerBase
        self._plotter = None

    def run(self):
        pass

    def add_solution(self, solution):
        self._solutions.append(solution)
