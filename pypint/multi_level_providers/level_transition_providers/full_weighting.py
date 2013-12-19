# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_level_transition_provider import ILevelTransitionProvider
import numpy as np
from pypint.utilities import critical_assert
from pypint import LOG


class FullWeighting(ILevelTransitionProvider):
    """
    Summary
    -------
    Full weighting restriction and prolongation.

    Extended Summary
    ----------------
    Full weighting restringates a fine level with :math:`n` points onto a
    coarse level with :math:`\\frac{n+1}{2}` points.

    Raises
    ------
    ValueError
        if number of fine level points is even (``fine_level_points``)

    Notes
    -----
    The prolongation is equal to the injective prolongation, where the intermedia fine points
    are calculated as the arithmetic mean of the sourounding coarse points.

    See Also
    --------
    .Injection
        Same prolongation operator.
    """
    def __init__(self, num_fine_points, num_coarse_points=-1):
        critical_assert(num_fine_points % 2 != 0,
                        ValueError, "Number of fine level points needs to be odd: {:d}".format(num_fine_points),
                        self)
        super(self.__class__, self).__init__(num_fine_points, num_coarse_points)
        self._n_coarse_points = int((self.num_fine_points + 1) / 2)
        self.restringation_operator = \
            np.zeros([self.num_coarse_points, self.num_fine_points])
        self.prolongation_operator = \
            np.zeros([self.num_fine_points, self.num_coarse_points])
        self._construct_transform_matrices()
        LOG.debug("Restringation operator: {:s}"
                  .format(self.restringation_operator))
        LOG.debug("Prolongation operator: {:s}"
                  .format(self.prolongation_operator))

    def prolongate(self, coarse_data):
        super(self.__class__, self).prolongate(coarse_data)
        return np.dot(self.prolongation_operator, coarse_data.transpose()) / 2.0

    def restringate(self, fine_data):
        super(self.__class__, self).restringate(fine_data)
        return np.dot(self.restringation_operator, fine_data.T) / 4.0

    def _construct_transform_matrices(self):
        # construct restringation operator
        for coarse in range(0, self.num_coarse_points):
            if coarse == 0:
                self.restringation_operator[0][0] = 2
                self.restringation_operator[0][1] = 1
            elif coarse == self.num_coarse_points - 1:
                self.restringation_operator[coarse][-2] = 1
                self.restringation_operator[coarse][-1] = 2
            else:
                fine = (2 * (coarse + 1)) - 2
                self.restringation_operator[coarse][fine - 1] = 1
                self.restringation_operator[coarse][fine] = 2
                self.restringation_operator[coarse][fine + 1] = 1

        # construct prolongation operator
        self.prolongation_operator = self.restringation_operator.copy().transpose()
        for fine in range(0, self.num_fine_points):
            if fine % 2 == 1:
                coarse = int((fine + 1) / 2)
                self.prolongation_operator[fine][coarse - 1] = 1.0
                self.prolongation_operator[fine][coarse] = 1.0
