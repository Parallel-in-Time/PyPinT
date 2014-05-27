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
        super(Constant, self).__init__(*args, **kwargs)
        HasExactSolutionMixin.__init__(self, *args, **kwargs)
        self.time_start = 0.0
        self.time_end = 1.0
        self.initial_value = shift * np.ones(self.dim_for_time_solver)
        self.constant = constant
        self._exact_function = lambda t: self.initial_value + self.constant * t
        self._strings['rhs_wrt_time'] = "C"
        self._strings['exact'] = 'u_0 + C * t'

    def evaluate_wrt_time(self, time, phi_of_time, **kwargs):
        super(Constant, self).evaluate_wrt_time(time, phi_of_time, **kwargs)
        return self.constant * np.ones(self.dim_for_time_solver)

    def print_lines_for_log(self):
        _lines = super(Constant, self).print_lines_for_log()
        _lines.update(HasExactSolutionMixin.print_lines_for_log(self))
        _lines['Coefficients'] = 'C = {:.3f}'.format(self.constant)
        return _lines

    def __str__(self):
        _outstr = super(Constant, self).__str__()
        _outstr += r", C={}".format(self.constant)
        _outstr += HasExactSolutionMixin.__str__(self)
        return _outstr
