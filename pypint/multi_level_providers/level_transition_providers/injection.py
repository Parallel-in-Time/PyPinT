# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_level_transition_provider import ILevelTransitionProvider
import numpy as np
from pypint.utilities import *
from pypint import LOG


class Injection(ILevelTransitionProvider):
    """
    Summary
    -------
    Injective restringation and prolongation.

    Extended Summary
    ----------------
    Injection restringates a fine level with :math:`n` points onto a
    coarse level with :math:`\\frac{n+1}{2}` points by leaving out every other
    data point.

    On prolongation, injection interpolates a new point between two coarse
    data points as their arithmetic mean.

    Parameters
    ----------
    fine_level_points : integer
        Number of points of the fine level.
        Must be odd.

    Raises
    ------
    ValueError
        if number of fine level points is even (``fine_level_points``)

    Notes
    -----
    Injective restringation only works for fine levels with an odd number of
    points.
    Injective prolongation only works for coarse levels with an even number of
    points.

    In addition, injection should only be used when the number of coarse points
    is a subset of the fine points.
    """
    def __init__(self, fine_level_points):
        super(self.__class__, self).__init__()
        if fine_level_points % 2 == 0:
            raise ValueError(func_name() +
                             "Number of fine level points needs to be odd: {:d}"
                             .format(fine_level_points))
        self._n_points = fine_level_points
        self._restringation_operator = \
            np.zeros([self.num_coarse_points, self.num_fine_points])
        self._prolongation_operator = \
            np.zeros([self.num_fine_points, self.num_coarse_points])
        self._construct_transform_matrices()
        LOG.debug("Restringation operator: {:s}"
                 .format(self._restringation_operator))
        LOG.debug("Prolongation operator: {:s}"
                 .format(self._prolongation_operator))

    def prolongate(self, coarse_data):
        super(self.__class__, self).prolongate(coarse_data)
        if not isinstance(coarse_data, np.ndarray):
            raise ValueError(func_name() +
                             "Given coarse data is not a numpy.ndarray: {:s}"
                             .format(type(coarse_data)))
        if coarse_data.size != self.num_coarse_points:
            raise ValueError(func_name() +
                             "Given coarse data is of wrong size: {:d}"
                             .format(coarse_data.size))
        return np.dot(self._prolongation_operator, coarse_data.transpose())

    def restringate(self, fine_data):
        super(self.__class__, self).restringate(fine_data)
        if not isinstance(fine_data, np.ndarray):
            raise ValueError(func_name() +
                             "Given fine data is not a numpy.ndarray: {:s}"
                             .format(type(fine_data)))
        if fine_data.size != self.num_fine_points:
            raise ValueError(func_name() +
                             "Given fine data is of wrong size: {:d}"
                             .format(fine_data.size))
        return np.dot(self._restringation_operator, fine_data.transpose())

    @property
    def num_fine_points(self):
        """
        Summary
        -------
        Accessor for the number of points of the fine level.

        Returns
        -------
        number of fine points : integer
            Number of points on the fine level.
        """
        return int(self._n_points)

    @property
    def num_coarse_points(self):
        """
        Summary
        -------
        Accessor for the number of points of the coarse level.

        Extended Summary
        ----------------
        The number of coarse points equals :math:`\\frac{n_{fine}+1}{2}`.

        Returns
        -------
        number of coarse points : integer
            Number of points on the fine level.
        """
        return int((self.num_fine_points + 1) / 2)

    def _construct_transform_matrices(self):
        # construct restringation operator
        for coarse in range(0, self.num_coarse_points):
            fine = (2 * (coarse + 1)) - 2
            self._restringation_operator[coarse][fine] = 1

        # construct prolongation operator
        self._prolongation_operator = \
            self._restringation_operator.copy().transpose()
        for fine in range(0, self.num_fine_points):
            if fine % 2 == 1:
                coarse = int((fine + 1) / 2)
                self._prolongation_operator[fine][coarse - 1] = 1.0 / 2.0
                self._prolongation_operator[fine][coarse] = 1.0 / 2.0
