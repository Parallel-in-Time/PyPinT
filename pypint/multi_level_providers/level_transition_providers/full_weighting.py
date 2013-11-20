# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_level_transition_provider import ILevelTransitionProvider
import numpy as np
from pypint.utilities import *


class FullWeighting(ILevelTransitionProvider):
    """
    Summary
    -------
    Full weighting restriction and prolongation.

    Extended Summary
    ----------------
    Full weighting restringates a fine level with :math:`n` points onto a
    coarse level with :math:`\frac{n+1}{2}` points.

    Parameters
    ----------
    fine_level_points : integer
        Number of points of the fine level.
    """
    def __init__(self, fine_level_points):
        super(self.__class__, self).__init__()
        self._n_points = fine_level_points
        self.restringation_operator = \
            np.zeros([self.num_coarse_points, self.num_fine_points])
        self.prolongation_operator = \
            np.zeros([self.num_fine_points, self.num_coarse_points])
        self._construct_transform_matrices()

    def prolongate(self, coarse_data):
        pass

    def restringate(self, fine_data):
        if not isinstance(fine_data, np.ndarray):
            raise ValueError(func_name() +
                             "Given fine data is not a numpy.ndarray: {:s}"
                             .format(type(fine_data)))
        return np.dot(self.restringation_operator, fine_data.T) / 4.0

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
        return self._n_points

    @property
    def num_coarse_points(self):
        """
        Summary
        -------
        Accessor for the number of points of the coarse level.

        Returns
        -------
        number of coarse points : integer
            Number of points on the fine level.
        """
        return (self.num_fine_points + 1) / 2

    def _construct_transform_matrices(self):
        # construct restringation operator
        for coarse in range(0, self.num_coarse_points):
            if coarse == 0:
                self.restringation_operator[0][0] = 2
                self.restringation_operator[0][1] = 1
            elif coarse == self.num_coarse_points - 1:
                if self.num_fine_points % 2 == 0:
                    self.restringation_operator[coarse][-3] = 1
                    self.restringation_operator[coarse][-2] = 2
                    self.restringation_operator[coarse][-1] = 1
                else:
                    self.restringation_operator[coarse][-2] = 1
                    self.restringation_operator[coarse][-1] = 2
            else:
                fine = (2 * coarse) - 1
                self.restringation_operator[coarse][fine - 1] = 1
                self.restringation_operator[coarse][fine] = 2
                self.restringation_operator[coarse][fine + 1] = 1

        # construct prolongation operator
#        for fine in range(0, self.num_fine_points):
#            pass
        raise NotImplementedError(func_name() +
                                  "Construction of prolongation operator " +
                                  "not yet implemented.")
