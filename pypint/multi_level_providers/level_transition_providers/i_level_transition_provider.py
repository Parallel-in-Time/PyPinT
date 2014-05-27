# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np

from pypint.utilities import assert_is_instance, assert_condition


class ILevelTransitionProvider(object):
    """Interface for level transition providers.
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        num_fine_points : :py:class:`int`
            *(optional)*
            Number of points of the fine level.

        num_coarse_points : :py:class:`int`
            *(optional)*
            Number of points of the coarse level.
        """
        self._prolongation_operator = None
        self._restringation_operator = None
        self._n_fine_points = int(kwargs['num_fine_points']) if 'num_fine_points' in kwargs else -1
        self._n_coarse_points = int(kwargs['num_coarse_points']) if 'num_coarse_points' in kwargs else -1

    def prolongate(self, coarse_data):
        """Prolongates given data from the coarse to the fine level.

        Parameters
        ----------
        coarse_data : :py:class:`numpy.ndarray`
            Coarse data vector to prolongate.

        Returns
        -------
        prolongated data : :py:class:`numpy.ndarray`
            Prolongated data on the fine level.

        Raises
        ------
        ValueError

            * if ``coarse_data`` is not a :py:class:`numpy.ndarray`
            * if ``coarse_data`` has more or less entries than :py:attr:`.num_coarse_points`
        """
        assert_is_instance(coarse_data, np.ndarray, descriptor="Coarse Data", checking_obj=self)
        assert_condition(coarse_data.shape[0] == self.num_coarse_points,
                         ValueError, message="Coarse Data is of wrong size: NOT {}".format(coarse_data.shape),
                         checking_obj=self)

    def restringate(self, fine_data):
        """Restringates given data from the fine to the coarse level.

        Parameters
        ----------
        fine_data : :py:class:`numpy.ndarray`
            Fine data vector to restringate.

        Returns
        -------
        restringated data : :py:class:`numpy.ndarray`
            Restringated data on the coarse level.

        Raises
        ------
        ValueError

            * if ``fine_data`` is not a `:py:class:`numpy.ndarray`
            * if ``fine_data`` has more or less entries than :py:attr:`.num_fine_points`
        """
        assert_is_instance(fine_data, np.ndarray, descriptor="Fine Data", checking_obj=self)
        assert_condition(fine_data.shape[0] == self.num_fine_points,
                         ValueError, message="Fine Data is of wrong size: {} != {}".format(fine_data.shape, self.num_fine_points),
                         checking_obj=self)

    @property
    def prolongation_operator(self):
        """Accessor for the prolongation operator.

        Parameters
        ----------
        prolongation_operator : :py:class:`numpy.ndarray`
            New prolongation operator to be used.

        Returns
        -------
        prolongation operator : :py:class:`numpy.ndarray`
            Current prolongation operator.
        """
        return self._prolongation_operator

    @prolongation_operator.setter
    def prolongation_operator(self, prolongation_operator):
        self._prolongation_operator = prolongation_operator

    @property
    def restringation_operator(self):
        """Accessor for the restringation operator.

        Parameters
        ----------
        restringation_operator : :py:class:`numpy.ndarray`
            New restringation operator to be used.

        Returns
        -------
        restringation operator : :py:class:`numpy.ndarray`
            Current restringation operator.
        """
        return self._restringation_operator

    @restringation_operator.setter
    def restringation_operator(self, restringation_operator):
        self._restringation_operator = restringation_operator

    @property
    def num_fine_points(self):
        """Accessor for the number of points of the fine level.

        Returns
        -------
        number of fine points : :py:class:`int`
            Number of points on the fine level.
        """
        return self._n_fine_points

    @property
    def num_coarse_points(self):
        """Accessor for the number of points of the coarse level.

        The number of coarse points equals :math:`\\frac{n_{fine}+1}{2}`.

        Returns
        -------
        number of coarse points : :py:class:`int`
            Number of points on the fine level.
        """
        return self._n_coarse_points

    def __str__(self):
        return "ITransitionProvider<0x%x>(fine_nodes=%d, coarse_nodes=%d)" \
               % (id(self), self._n_fine_points, self._n_coarse_points)
