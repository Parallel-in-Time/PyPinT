# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.plugins.loggers import *

class IAnalyzer(object):
    def __init__(self):
        self.__solutions = []
        self.__logger = LoggerBase
        self.__plotter = None

    def run(self):
        pass

    def add_solution(self, solution):
        self.__solutions.append(solution)
