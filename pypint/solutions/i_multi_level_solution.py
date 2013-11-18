# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.solutions.i_solution import ISolution


class IMultiLevelSolution(ISolution):
    def __init__(self):
        super(self.__class__, self).__init__()
        self._used_levels = None

    @property
    def used_levels(self):
        return self._used_levels

    @used_levels.setter
    def used_levels(self, used_levels):
        self._used_levels = used_levels
