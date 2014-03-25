# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from copy import deepcopy

import numpy as np

from pypint.utilities import assert_is_instance, assert_condition


class IDiagnosisValue(object):
    """Storage and handler of diagnosis values of iterative time solvers.

    Comparability
        It can be equality-compared (i.e. operators ``==`` and ``!=`` are implemented).
        The other comparison operators such as ``<``, ``<=``, ``>`` and ``>=`` are not implemented as these do not make
        any sense for this type of container.

        Two instances are the same, if they have the same :py:attr:`.numeric_type` and their :py:attr:`.value` are the
        same with respect to :py:meth:`numpy.array_equal`.
    Hashable
        It is not hashable due to its wrapping around :py:class:`numpy.ndarray`.

    .. todo::
        Extend this interface to emulate a numeric type.
        This includes :py:meth:`.__add__`, :py:meth:`.__sub__`, etc.
    """

    def __init__(self, value):
        """

        Parameters
        ----------
        value : :py:class:`numpy.ndarray`

        Raises
        ------
        ValueError:
            If ``value`` is not a :py:class:`numpy.ndarray`.
        """
        assert_is_instance(value, np.ndarray, descriptor="Diagnosis Values", checking_obj=self)
        self._data = value
        self._numeric_type = self.value.dtype

    @property
    def value(self):
        """Read-only accessor for the value.

        Returns
        -------
        value : :py:class:`numpy.ndarray`
        """
        return self._data

    @property
    def numeric_type(self):
        """Read-only accessor for the numerical type of the value.

        The type is derived from the given values.

        Returns
        -------
        numeric_type : :py:class:`numpy.dtype`
        """
        return self._numeric_type

    def __copy__(self):
        copy = self.__class__.__new__(self.__class__)
        copy.__dict__.update(self.__dict__)
        return copy

    def __deepcopy__(self, memo):
        copy = self.__class__.__new__(self.__class__)
        memo[id(self)] = copy
        for item, value in self.__dict__.items():
            setattr(copy, item, deepcopy(value, memo))
        return copy

    def __eq__(self, other):
        assert_condition(isinstance(other, self.__class__), TypeError,
                         message="Can not compare {} with {}".format(self.__class__, other.__class__.__name__),
                         checking_obj=self)
        return (
            self.numeric_type == other.numeric_type
            and np.array_equal(self.value, other.value)
        )

    def __ge__(self, other):
        return NotImplemented

    def __gt__(self, other):
        return NotImplemented

    def __le__(self, other):
        return NotImplemented

    def __lt__(self, other):
        return NotImplemented

    def __ne__(self, other):
        assert_condition(isinstance(other, self.__class__), TypeError,
                         message="Can not compare {} with {}".format(self.__class__, other.__class__.__name__),
                         checking_obj=self)
        return not self.__eq__(other)

    __hash__ = None


__all__ = ['IDiagnosisValue']
