# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class MultiLevelProvider(object):
    def __init__(self):
        self._level_transition_providers = []
        self._level_integrators = []
        self._num_levels = -1

    def integrator(self, level):
        return self.level_integrators[level]

    def prolongate(self, coarse_level, fine_level, coarse_data):
        return self._level_transition(coarse_level=coarse_level).prolongate(coarse_data)

    def restringate(self, fine_level, coarse_level, fine_data):
        return self._level_transition(fine_level=fine_level).restringate(fine_data)

    def __level_transition(self, coarse_level=None, fine_level=None):
        return self.level_transitions_providers[coarse_level or fine_level-1]

    @property
    def num_levels(self):
        return self._num_levels

    @num_levels.setter
    def num_levels(self, num_levels):
        self._num_levels = num_levels
