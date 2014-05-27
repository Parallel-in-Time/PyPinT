# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.multi_level_providers.multi_level_provider import MultiLevelProvider
from pypint.multi_level_providers.level_transition_providers.time_transition_provider import TimeTransitionProvider
from pypint.utilities import assert_is_instance


class MultiTimeLevelProvider(MultiLevelProvider):
    def __init__(self, num_levels=0, default_integrator=None, default_transitioner=None):
        """
        Parameters
        ----------
        num_levels : :py:class:`int`
            Number of initial levels.

        default_transitioner : :py:class:`.ILevelTransitionProvider`
            Default level transitioner to be used for all level transitions unless a specific one is specified.
        """
        super(MultiTimeLevelProvider, self).__init__(num_levels=num_levels,
                                                     default_integrator=default_integrator,
                                                     default_transitioner=default_transitioner)

    def add_level_transition(self, transitioner, coarse_level, fine_level):
        """Adds specialized level transitioner for specified levels.

        Parameters
        ----------
        transitioner : :py:class:`.TimeTransitionProvider`
            Special level transitioner for specified prolongation and restringation between given coarse and fine level.

        coarse_level : :py:class:`int`
            Coarse level of the transitioner.

        fine_level : :py:class:`int`
            Fine level of the transitioner.

        Raises
        ------
        ValueError
            if ``transitioner`` is not an :py:class:`.TimeTransitionProvider`
        """
        assert_is_instance(transitioner, TimeTransitionProvider, descriptor="Level Transitioner", checking_obj=self)

        # extend/initialize level_transition_provider map if necessary
        if coarse_level not in self._level_transitioners:
            self._level_transitioners[coarse_level] = {}

        self._level_transitioners[coarse_level][fine_level] = transitioner

    def __str__(self):
        return "MultiTimeLevelProvider<0x%x>(num_level=%d)" % (id(self), self.num_levels)


__all__ = ['MultiTimeLevelProvider']
