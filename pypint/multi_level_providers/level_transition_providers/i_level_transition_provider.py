# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

import numpy as np
from pypint.utilities import func_name


class ILevelTransitionProvider(object):
    """
    Summary
    -------
    Interface for level transition providers.
    """
    def __init__(self):
        self._prolongation_operator = None
        self._restringation_operator = None

    def prolongate(self, coarse_data):
        if not isinstance(coarse_data, np.ndarray):
            raise ValueError(func_name() +
                             "Given coarse data is not a numpy.ndarray: {:s}"
                             .format(type(coarse_data)))
        if coarse_data.size != self.num_coarse_points:
            raise ValueError(func_name() +
                             "Given coarse data is of wrong size: {:d}"
                             .format(coarse_data.size))

    def restringate(self, fine_data):
        if not isinstance(fine_data, np.ndarray):
            raise ValueError(func_name() +
                             "Given fine data is not a numpy.ndarray: {:s}"
                             .format(type(fine_data)))
        if fine_data.size != self.num_fine_points:
            raise ValueError(func_name() +
                             "Given fine data is of wrong size: {:d}"
                             .format(fine_data.size))

    @property
    def prolongation_operator(self):
        return self._prolongation_operator

    @prolongation_operator.setter
    def prolongation_operator(self, prolongation_operator):
        self._prolongation_operator = prolongation_operator

    @property
    def restringation_operator(self):
        return self._restringation_operator

    @restringation_operator.setter
    def restringation_operator(self, restringation_operator):
        self._restringation_operator = restringation_operator

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
        The number of coarse points equals :math:`\frac{n_{fine}+1}{2}`.

        Returns
        -------
        number of coarse points : integer
            Number of points on the fine level.
        """
        return int((self.num_fine_points + 1) / 2)
