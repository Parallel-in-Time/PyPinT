# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np

from pypint.problems.i_problem import IProblem
from pypint.utilities import assert_condition, assert_is_instance


class IInitialValueProblem(IProblem):
    """Basic interface for initial value problems.

    Parameters
    ----------
    initial_value : :py:class:`numpy.ndarray`
        Initial value of :math:`u(t_0,\\phi(t_0))` with :math:`t_0` being the time interval start.
    """
    def __init__(self, *args, **kwargs):
        super(IInitialValueProblem, self).__init__(*args, **kwargs)

        self._initial_value = None
        if "initial_value" in kwargs:
            self.initial_value = kwargs["initial_value"]

    @property
    def initial_value(self):
        """Accessor for the initial value.

        Parameters
        ----------
        initial value : :py:class:`numpy.ndarray`
            Initial value of the solution.

        Returns
        -------
        initial value : :py:class:`numpy.ndarray`
            Initial value of the solution.

        Raises
        ------
        ValueError

            * if ``initial_value`` is not a :py:class:`numpy.ndarray`
            * if ``initial_value``'s size is not equal the number of spacial :py:attr:`.dim`
        """
        return self._initial_value

    @initial_value.setter
    def initial_value(self, initial_value):
        assert_is_instance(initial_value, np.ndarray, descriptor="Initial Value", checking_obj=self)
        assert_condition(initial_value.size == self.dim, ValueError,
                         message="Initial value must match spacial dimension: {:d} != {:d}"
                                 .format(self.dim, initial_value.size),
                         checking_obj=self)
        self._initial_value = initial_value

    def print_lines_for_log(self):
        _lines = super(IInitialValueProblem, self).print_lines_for_log()
        _lines['Initial Value'] = 'u({:.3f}) = {}'.format(self.time_start, self.initial_value)
        return _lines

    def __str__(self):
        _out = super(IInitialValueProblem, self).__str__()
        _out += r", u({:.2f})={:s}".format(self.time_start, self.initial_value)
        return _out


__all__ = ['IInitialValueProblem']
