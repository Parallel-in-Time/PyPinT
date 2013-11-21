# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.integrators.integrator_base import IntegratorBase
from .level_transition_providers.i_level_transition_provider \
    import ILevelTransitionProvider
from pypint.utilities import *


class MultiLevelProvider(object):
    def __init__(self, num_levels=0, default_integrator=None,
                 default_transitioner=None):
        """
        Paramters
        ---------
        num_levels : integer
            Number of initial levels.
        default_transitioner : :py:class:`.ILevelTransitionProvider`
            Default level transitioner to be used for all level transitions
            unless a specific one is specified.
        """
        self._level_transitioners = {}
        self._default_transitioner = default_transitioner
        self._level_integrators = []
        self._num_levels = num_levels
        if num_levels is not None and default_integrator is not None:
            for level in range(0, num_levels):
                self.add_coarse_level(default_integrator)

    def integrator(self, level):
        """
        Summary
        -------
        Accessor for the integrator for the specified level.

        Parameters
        ----------
        level : integer
            Level to retrieve integrator for.

        Returns
        -------
        integrator : :py:class:`.IntegratorBase`
            Stored integrator for given level.
        """
        return self._level_integrators[level]

    def prolongate(self, coarse_data, coarse_level, fine_level=None):
        """
        Summary
        -------
        Prolongates given data from coarser to finer level.

        Parameters
        ----------
        coarse_data : numpy.ndarray
            Coarse data to prolongate.
        coarse_level : integer
            Level of the given data to prolongate from.
        fine_level : integer (optional)
            Fine level to prolongate onto. In case it is ``None`` the next
            finer level is taken.

        Returns
        -------
        prolongated data : numpy.ndarray
            The prolongated data.

        See Also
        --------
        .ILevelTransitionProvider.prolongate
            For details on prolongation.
        """
        return self._level_transition(coarse_level=coarse_level,
                                      fine_level=fine_level)\
                   .prolongate(coarse_data)

    def restringate(self, fine_data, fine_level, coarse_level=None):
        """
        Summary
        -------
        Restringates given data from finer to coarser level.

        Parameters
        ----------
        fine_data : numpy.ndarray
            Fine data to restringate.
        fine_level : integer
            Level of the given data to restringate from.
        coarse_level : integer (optional)
            Coarse level to restringate onto. In case it is ``None`` the next
            coarser level is taken.

        Returns
        -------
        restringated data : numpy.ndarray
            The restringated data.

        See Also
        --------
        .ILevelTransitionProvider.restringate
            For details on restringation.
        """
        return self._level_transition(coarse_level=coarse_level,
                                      fine_level=fine_level)\
                   .restringate(fine_data)

    def add_coarse_level(self, integrator, top_level=-1):
        """
        Summary
        -------
        Adds a coarser level including an integrator and transitioner.

        Parameters
        ----------
        integrator : :py:class:`.IntegratorBase`
            Integrator for the new level.
        top_level : integer
            Next finer level of the new level.
            ``0`` is the finest level, ``-1`` the currently coarsest.

        Raises
        ------
        ValueError
            If ``integrator`` is not an :py:class:`.IntegratorBase`.
        """
        if not isinstance(integrator, IntegratorBase):
            raise ValueError(func_name(self) +
                             "Integrator is of invalid type: {:s}"
                             .format(type(integrator)))
        self._num_levels += 1
        self._level_integrators.insert(top_level, integrator)

    def add_level_transition(self, transitioner, coarse_level, fine_level):
        """
        Summary
        -------
        Adds specialized level transitioner for specified levels.

        Parameters
        ----------
        transitioner : :py:class:`.ILevelTransitionProvider`
            Special level transitioner for specified prolongation and
            restringation between given coarse and fine level.
        coarse_level : integer
            Coarse level of the transitioner.
        fine_level : integer
            Fine level of the transitioner.

        Raises
        ------
        ValueError
            if ``transitioner`` is not an :py:class:`.ILevelTransitionProvider`
        """
        if not isinstance(transitioner, ILevelTransitionProvider):
            raise ValueError(func_name(self) +
                             "Level transitioner is of invalid type: {:s}"
                             .format(type(transitioner)))

        # extend/initialize level_transition_provider map if necessary
        if coarse_level not in self._level_transitioners:
            self._level_transitioners[coarse_level] = {}

        self._level_transitioners[coarse_level][fine_level] = transitioner

    @property
    def num_levels(self):
        """
        Summary
        -------
        Accessor for the number of levels.

        Returns
        -------
        num_levels : integer
            Number of levels of this Multi-Level Provider.
        """
        return self._num_levels

    def _level_transition(self, coarse_level=None, fine_level=None):
        """
        Summary
        -------
        Extracts level transition provider for given coarse and fine levels.

        Parameters
        ----------
        coarse_level : integer
            Coarse level of the level transitioner.
        fine_level : integer
            Fine level of the level transitioner.

        Returns
        -------
        level_transitioner : :py:class:`.ILevelTransitionProvider`
            Level transition provider to restringate and prolongate between
            the given coarse and fine level.
            In case no specialized transitioner is found, the default one is
            returned.

        Raises
        ------
        ValueError
            * if ``coarse_level`` and ``fine_level`` are ``None``
            * if ``fine_level`` is ``None`` and ``coarse_level`` is the finest
              one
            * if ``coarse_level`` is ``None`` and ``fine_level`` is the
              coarsest one
        """
        if coarse_level is None and fine_level is None:
            raise ValueError(func_name(self) +
                             "Either coarse or fine level index must be given")
        if fine_level is None:
            fine_level = coarse_level - 1
        if coarse_level is None:
            coarse_level = fine_level + 1
        if fine_level < 0:
            raise ValueError(func_name(self) +
                             "There is no finer level than given coarse one: {:d}"
                             .format(coarse_level))
        if coarse_level >= self.num_levels:
            raise ValueError(func_name(self) +
                             "There is no coarser level than given fine one: {:d}"
                             .format(fine_level))

        if coarse_level in self._level_transitioners \
                and fine_level in self._level_transitioners[coarse_level]:
            return self._level_transitioners[coarse_level][fine_level]
        else:
            return self._default_transitioner
