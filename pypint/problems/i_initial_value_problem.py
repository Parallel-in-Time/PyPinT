# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_problem import IProblem
from ..utilities import assert_condition, assert_is_instance
import numpy as np


class IInitialValueProblem(IProblem):
    """
    Summary
    -------
    Basic interface for initial value problems.

    Parameters
    ----------
    In addition to the ones of :py:class:`.IProblem` the following options are supported:

    initial_value : float | ``numpy.ndarray``
        Initial value of :math:`u(t_0,\\phi(t_0))` with :math:`t_0` being the time interval
        start.
    """
    def __init__(self, *args, **kwargs):
        super(IInitialValueProblem, self).__init__(*args, **kwargs)

        self._initial_value = None
        if "initial_value" in kwargs:
            self.initial_value = kwargs["initial_value"]

    @property
    def initial_value(self):
        """
        Summary
        -------
        Accessor for the initial value.

        Parameters
        ----------
        initial value : float | numpy.ndarray
            Initial value of the solution.

        Returns
        -------
        initial value : float | numpy.ndarray
            Initial value of the solution.
        """
        return self._initial_value

    @initial_value.setter
    def initial_value(self, initial_value):
        assert_is_instance(initial_value, np.ndarray,
                           "Initial value must be a numpy.ndarray: NOT {:s}".format(initial_value.__class__.__name__),
                           self)
        assert_condition(initial_value.size == self.dim,
                         ValueError, "Initial value must match spacial dimension: {:d} != {:d}"
                                     .format(self.dim, initial_value.size),
                         self)
        self._initial_value = initial_value

    def __str__(self):
        _out = super(IInitialValueProblem, self).__str__()
        _out += r", u({:.2f})={:s}".format(self.time_start, self.initial_value)
        return _out
