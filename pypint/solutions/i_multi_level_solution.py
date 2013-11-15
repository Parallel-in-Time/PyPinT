# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.solutions.i_solution import ISolution


class IMultiLevelSolution(ISolution):
    def __init__(self):
        self.__used_levels = None
        super().__init__()

    @property
    def used_levels(self):
        return self.__used_levels

    @used_levels.setter
    def used_levels(self, used_levels):
        self.__used_levels = used_levels
