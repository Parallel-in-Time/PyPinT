# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from copy import deepcopy

import numpy as np

from pypint.solvers.diagnosis import Error, Residual
from pypint.utilities import assert_is_instance, assert_condition, class_name


class StepSolutionData(object):
    """Storage for the solution of a single time point.

    Finalization
        The attributes (:py:attr:`.value`, :py:attr:`.time_point`, :py:attr:`.error` and :py:attr:`.residual` can only
        be modified before calling :py:meth:`.finalize`.
        It is not intended and possible to *definalize* a once finalized :py:class:`StepSolutionData` instance.

    Comparability
        It can be compared with respect to all numerical comparison operators.
        For all operators to evaluate to :py:class:`True` it is necessary that :py:attr:`.dim` and
        :py:attr:`.numeric_type` are the same.
        Equality is given, if :py:attr:`.time_point`, :py:attr:`.error` and :py:attr:`.residual` are the same as well
        as :py:attr:`.value` with respect to :py:meth:`numpy.array_equal`.
        The other comparison operators do not take :py:attr:`.value`, :py:attr:`.error` and :py:attr:`.residual` into
        account and induce an order only with respect to :py:attr:`.time_point`.

    Hashable
        It is not hashable due to its wrapping around :py:class:`numpy.ndarray`.
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        value : :py:class:`numpy.ndarray`
            Solution value.
            Size of the vector must equal the spacial dimension of the problem.

        time_point : :py:class:`float`
            Time point of associated with the solution.

        error : :py:class:`.Error` or :py:class:`numpy.ndarray`
            Error of the solution.

        residual : :py:class:`.Residual` or :py:class:`numpy.ndarray`
            Residual of the solution.
            Same abundance of constrains apply as for the error.

        Raises
        ------
        ValueError :

            * if ``value`` is not a :py:class:`numpy.ndarray`
            * if ``time_point`` is not a :py:class:`float`
            * if either :py:attr:`.error` or :py:attr:`.residual` raises

        UserWarning :
            if no ``value`` or ``time_point`` is given

        Notes
        -----
        The spacial dimension and the numerical type are derived from the given solution values.
        Thus, specifying :py:attr:`.ISolutionData.dim` and :py:attr:`.ISolutionData.numeric_type` is not recommended
        (in fact they are ignored).
        """
        self._data = None
        self._time_point = None
        self._error = None
        self._residual = None

        self._dim = 0
        self._numeric_type = None
        self._finalized = False

        if 'value' in kwargs:
            self.value = kwargs['value']

        if 'time_point' in kwargs:
            self.time_point = kwargs['time_point']

        if 'error' in kwargs:
            self.error = kwargs['error']

        if 'residual' in kwargs:
            self.residual = kwargs['residual']

    def finalize(self):
        """Locks this storage data instance.

        Raises
        ------
        ValueError :
            If it has already been locked.
        """
        assert_condition(not self.finalized, AttributeError,
                         message="This solution data storage is already finalized.", checking_obj=self)
        self._finalized = True

    @property
    def finalized(self):
        """Accessor for the lock state.

        Returns
        -------
        finilized : :py:class:`bool`

            :py:class:`True`
                if it has been finalized before
            :py:class:`False`
                otherwise
        """
        return self._finalized

    @property
    def value(self):
        """Accessor for the solution value.

        Returns
        -------
        value : :py:class:`numpy.ndarray`

        Raises
        ------
        ValueError :
            If this storage data instance has been finalized.
            *(only setter)*
        """
        return self._data

    @value.setter
    def value(self, value):
        assert_condition(not self.finalized, AttributeError,
                         message="Cannot change this solution data storage any more.", checking_obj=self)
        assert_is_instance(value, np.ndarray, descriptor="Values", checking_obj=self)
        self._dim = value.size
        self._numeric_type = value.dtype
        self._data = value

    @property
    def time_point(self):
        """Accessor for the associated time point.

        Parameters
        ----------
        time_point : :py:class:`float`

        Returns
        -------
        time_point : :py:class:`float`

        Raises
        ------
        ValueError :
            If this storage data instance has been finalized.
            *(only setter)*
        """
        return self._time_point

    @time_point.setter
    def time_point(self, time_point):
        assert_condition(not self.finalized, AttributeError,
                         message="Cannot change this solution data storage any more.", checking_obj=self)
        assert_is_instance(time_point, float, descriptor="Time Point", checking_obj=self)
        self._time_point = time_point

    @property
    def error(self):
        """Accessor for the error.

        Parameters
        ----------
        error : :py:class:`.Error` or :py:class:`numpy.ndarray`

        Returns
        -------
        error : :py:class:`.Error`
            or :py:class:`None` if no error is given

        Raises
        ------
        ValueError :
            If this storage data instance has been finalized.
            *(only setter)*
        """
        return self._error

    @error.setter
    def error(self, error):
        assert_condition(not self.finalized, AttributeError,
                         message="Cannot change this solution data storage any more.", checking_obj=self)
        assert_is_instance(error, (np.ndarray, Error), descriptor="Error", checking_obj=self)
        self._error = error if isinstance(error, Error) else Error(value=error)

    @property
    def residual(self):
        """Accessor for the residual.

        Parameters
        ----------
        residual : :py:class:`.Residual` or :py:class:`numpy.ndarray`

        Returns
        -------
        residual : :py:class:`.Residual`
            or :py:class:`None` if no residual is given

        Raises
        ------
        ValueError :
            If this storage data instance has been finalized.
            *(only setter)*
        """
        return self._residual

    @residual.setter
    def residual(self, residual):
        assert_condition(not self.finalized, AttributeError,
                         message="Cannot change this solution data storage any more.", checking_obj=self)
        assert_is_instance(residual, (np.ndarray, Residual), descriptor="Residual", checking_obj=self)
        self._residual = residual if isinstance(residual, Residual) else Residual(value=residual)

    @property
    def dim(self):
        """Read-only accessor for the spacial dimension.

        Returns
        -------
        dim : :py:class:`int`
        """
        return self._dim

    @property
    def numeric_type(self):
        """Read-only accessor for the numerical type.

        Returns
        -------
        numeric_type : :py:class:`numpy.dtype`
        """
        return self._numeric_type

    def __str__(self):
        return "StepSolutionData(value={}, time_point={}, finalized={})"\
            .format(self.value, self.time_point, self.finalized)

    def __copy__(self):
        copy = self.__class__.__new__(self.__class__)
        copy.__dict__.update(self.__dict__)
        copy._finalized = False
        return copy

    def __deepcopy__(self, memo):
        copy = self.__class__.__new__(self.__class__)
        memo[id(self)] = copy
        for item, value in self.__dict__.items():
            setattr(copy, item, deepcopy(value, memo))
        copy._finalized = False
        return copy

    def __eq__(self, other):
        assert_is_instance(other, StepSolutionData,
                           message="Can not compare StepSolutionData with {}".format(class_name(other)),
                           checking_obj=self)
        return (
            self.time_point == other.time_point
            and self.dim == other.dim
            and self.numeric_type == other.numeric_type
            and np.array_equal(self.value, other.value)
            and self.error == other.error
            and self.residual == other.residual
        )

    def __ge__(self, other):
        assert_is_instance(other, StepSolutionData,
                           message="Can not compare StepSolutionData with {}".format(class_name(other)),
                           checking_obj=self)
        return (
            self.dim == other.dim
            and self.numeric_type == other.numeric_type
            and self.time_point >= other.time_point
        )

    def __gt__(self, other):
        assert_is_instance(other, StepSolutionData,
                           message="Can not compare StepSolutionData with {}".format(class_name(other)),
                           checking_obj=self)
        return (
            self.dim == other.dim
            and self.numeric_type == other.numeric_type
            and self.time_point > other.time_point
        )

    def __le__(self, other):
        assert_is_instance(other, StepSolutionData,
                           message="Can not compare StepSolutionData with {}".format(class_name(other)),
                           checking_obj=self)
        return (
            self.dim == other.dim
            and self.numeric_type == other.numeric_type
            and self.time_point < other.time_point
        )

    def __lt__(self, other):
        assert_is_instance(other, StepSolutionData,
                           message="Can not compare StepSolutionData with {}".format(class_name(other)),
                           checking_obj=self)
        return (
            self.dim == other.dim
            and self.numeric_type == other.numeric_type
            and self.time_point < other.time_point
        )

    def __ne__(self, other):
        assert_is_instance(other, StepSolutionData,
                           message="Can not compare StepSolutionData with {}".format(class_name(other)),
                           checking_obj=self)
        return not self.__eq__(other)

    # StepSolutionData is mutable
    __hash__ = None


__all__ = ['StepSolutionData']
