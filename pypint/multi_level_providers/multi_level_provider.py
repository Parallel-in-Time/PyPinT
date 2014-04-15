# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.integrators.integrator_base import IntegratorBase
from pypint.multi_level_providers.level_transition_providers.i_level_transition_provider \
    import ILevelTransitionProvider
from pypint.utilities import assert_condition, assert_is_instance


class MultiLevelProvider(object):
    def __init__(self, num_levels=0, default_integrator=None,
                 default_transitioner=None):
        """
        Parameters
        ----------
        num_levels : :py:class:`int`
            Number of initial levels.

        default_transitioner : :py:class:`.ILevelTransitionProvider`
            Default level transitioner to be used for all level transitions unless a specific one is specified.
        """
        self._level_transitioners = {}
        self._default_transitioner = default_transitioner
        self._level_integrators = []
        self._num_levels = num_levels
        if num_levels is not None and default_integrator is not None:
            for level in range(0, num_levels):
                self.add_coarse_level(default_integrator)

    def integrator(self, level):
        """Accessor for the integrator for the specified level.

        Parameters
        ----------
        level : :py:class:`int`
            Level to retrieve integrator for.

        Returns
        -------
        integrator : :py:class:`.IntegratorBase`
            Stored integrator for given level.
        """
        return self._level_integrators[level]

    def prolongate(self, coarse_data, coarse_level, fine_level=None):
        """Prolongates given data from coarser to finer level.

        Parameters
        ----------
        coarse_data : :py:class:`numpy.ndarray`
            Coarse data to prolongate.

        coarse_level : :py:class:`int`
            Level of the given data to prolongate from.

        fine_level : :py:class:`int`
            *(optional)*
            Fine level to prolongate onto.
            In case it is :py:class:`None` the next finer level is taken.

        Returns
        -------
        prolongated data : :py:class:`numpy.ndarray`
            The prolongated data.

        See Also
        --------
        :py:meth:`.ILevelTransitionProvider.prolongate` : for details on prolongation
        """
        return self._level_transition(coarse_level=coarse_level, fine_level=fine_level).prolongate(coarse_data)

    def restringate(self, fine_data, fine_level, coarse_level=None):
        """Restringates given data from finer to coarser level.

        Parameters
        ----------
        fine_data : :py:class:`numpy.ndarray`
            Fine data to restringate.

        fine_level : :py:class:`int`
            Level of the given data to restringate from.

        coarse_level : :py:class:`int`
            *(optional)*
            Coarse level to restringate onto.
            In case it is :py:class:`None` the next coarser level is taken.

        Returns
        -------
        restringated data : :py:class:`numpy.ndarray`
            The restringated data.

        See Also
        --------
        :py:meth:`.ILevelTransitionProvider.restringate` : for details on restringation
        """
        return self._level_transition(coarse_level=coarse_level, fine_level=fine_level).restringate(fine_data)

    def add_coarse_level(self, integrator, top_level=0):
        """Adds a coarser level including an integrator and transitioner.

        Parameters
        ----------
        integrator : :py:class:`.IntegratorBase`
            Integrator for the new level.

        top_level : :py:class:`int`
            Next finer level of the new level.
            ``-1`` is the finest level, ``0`` the currently coarsest.

        Raises
        ------
        ValueError
            If ``integrator`` is not an :py:class:`.IntegratorBase`.
        """
        assert_is_instance(integrator, IntegratorBase, descriptor="Integrator", checking_obj=self)
        self._num_levels += 1
        self._level_integrators.insert(top_level, integrator)

    def add_level_transition(self, transitioner, coarse_level, fine_level):
        """Adds specialized level transitioner for specified levels.

        Parameters
        ----------
        transitioner : :py:class:`.ILevelTransitionProvider`
            Special level transitioner for specified prolongation and restringation between given coarse and fine level.

        coarse_level : :py:class:`int`
            Coarse level of the transitioner.

        fine_level : :py:class:`int`
            Fine level of the transitioner.

        Raises
        ------
        ValueError
            if ``transitioner`` is not an :py:class:`.ILevelTransitionProvider`
        """
        assert_is_instance(transitioner, ILevelTransitionProvider, descriptor="Level Transitioner", checking_obj=self)

        # extend/initialize level_transition_provider map if necessary
        if coarse_level not in self._level_transitioners:
            self._level_transitioners[coarse_level] = {}

        self._level_transitioners[coarse_level][fine_level] = transitioner

    @property
    def num_levels(self):
        """Accessor for the number of levels.

        Returns
        -------
        num_levels : :py:class:`int`
            Number of levels of this Multi-Level Provider.
        """
        return self._num_levels

    def _level_transition(self, coarse_level=None, fine_level=None):
        """Extracts level transition provider for given coarse and fine levels.

        Parameters
        ----------
        coarse_level : :py:class:`int`
            Coarse level of the level transitioner.

        fine_level : :py:class:`int`
            Fine level of the level transitioner.

        Returns
        -------
        level_transitioner : :py:class:`.ILevelTransitionProvider`
            Level transition provider to restringate and prolongate between the given coarse and fine level.
            In case no specialized transitioner is found, the default one is returned.

        Raises
        ------
        ValueError

            * if ``coarse_level`` and ``fine_level`` are :py:class:`None`
            * if ``fine_level`` is :py:class:`None` and ``coarse_level`` is the finest one
            * if ``coarse_level`` is :py:class:`None` and ``fine_level`` is the coarsest one
        """
        assert_condition(coarse_level is not None or fine_level is not None,
                         ValueError, message="Either coarse or fine level index must be given", checking_obj=self)
        if fine_level is None:
            fine_level = coarse_level + 1
        if coarse_level is None:
            coarse_level = fine_level - 1
        assert_condition(fine_level < self.num_levels, ValueError,
                         message="There is no finer level than given coarse one: {:d}".format(coarse_level),
                         checking_obj=self)
        assert_condition(coarse_level >= 0, ValueError,
                         message="There is no coarser level than given fine one: {:d}".format(fine_level),
                         checking_obj=self)

        if coarse_level in self._level_transitioners and fine_level in self._level_transitioners[coarse_level]:
            return self._level_transitioners[coarse_level][fine_level]
        else:
            return self._default_transitioner
