# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class IAnalyzer(object):
    def __init__(self):
        self._solutions = []
        self._plotter = None

    def run(self):
        pass

    def add_solution(self, solution):
        self._solutions.append(solution)
