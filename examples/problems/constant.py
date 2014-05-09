# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np

from pypint.problems import IInitialValueProblem, HasExactSolutionMixin


class Constant(IInitialValueProblem, HasExactSolutionMixin):
    """:math:`u'(t, \\phi_t) &= C`

    Describes the following first-order ODE initial value problem:

    .. math::

        \\begin{align}
            u'(t, \\phi_t) &= C \\\\
                   u(t, 0) &= IV
        \\end{align}

    With the exact solution:

    .. math::

        u(t, \\phi_t) = 1-IV\\phi_t

    Parameters
    ----------
    constant : :py:class:`float`
        Constant value :math:`C`

    shift : :py:class:`float`
        Initial value :math:`IV`, which is the shift along the :math:`y` axis.
    """
    def __init__(self, constant=-1.0, shift=1.0, *args, **kwargs):
        super(Constant, self).__init__(args, kwargs)
        self.time_start = 0.0
        self.time_end = 1.0
        self.initial_value = shift * np.ones(self.dim)
        self.constant = constant
        self._strings['rhs_wrt_time'] = "C"
        self._exact_function = lambda t: self.initial_value + self.constant * t

    def evaluate_wrt_time(self, time, phi_of_time, partial=None):
        super(Constant, self).evaluate_wrt_time(time, phi_of_time, partial)
        return self.constant * np.ones(self.dim)

    def print_lines_for_log(self):
        _lines = super(Constant, self).print_lines_for_log()
        _lines['Coefficients'] = 'C = {:.3f}'.format(self.constant)
        return _lines

    def __str__(self):
        str = super(Constant, self).__str__()
        str += r", C={}".format(self.constant)
        return str
