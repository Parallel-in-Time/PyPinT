# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np

from pypint.problems import IInitialValueProblem, HasExactSolutionMixin, HasDirectImplicitMixin
from pypint.utilities import assert_condition, assert_is_instance, class_name, assert_named_argument
from pypint.solvers.cores.implicit_sdc_core import ImplicitSdcCore
from pypint.solvers.cores.implicit_mlsdc_core import ImplicitMlSdcCore
from pypint.solvers.cores.semi_implicit_mlsdc_core import SemiImplicitMlSdcCore
from pypint.utilities.logging import LOG


class LambdaU(IInitialValueProblem, HasExactSolutionMixin, HasDirectImplicitMixin):
# class LambdaU(IInitialValueProblem, HasExactSolutionMixin):
    """:math:`u'(t, \\phi_t) = \\lambda u(t, \\phi_t)`

    Describes the following first-order ODE initial value problem:

    .. math::

        \\begin{align}
            u'(t, \\phi_t) &= \\lambda u(t, \\phi_t) \\\\
                   u(t, 0) &= 1
        \\end{align}

    With the exact solution:

    .. math::

        u(t, \\phi_t) = e^{\\lambda\\phi_t}

    Parameters
    ----------
    lmbda : :py:class:`float`
        *(optional)*
        Coefficient :math:`\\lambda`
    """
    def __init__(self, *args, **kwargs):
        super(LambdaU, self).__init__(*args, **kwargs)
        HasExactSolutionMixin.__init__(self, *args, **kwargs)
        HasDirectImplicitMixin.__init__(self, *args, **kwargs)
        if self.time_start is None:
            self.time_start = 0.0
        if self.time_end is None:
            self.time_end = 1.0
        if self.initial_value is None:
            self.initial_value = complex(1.0, 0.0) * np.ones(self.dim)
            # self.initial_value = 1.0 * np.ones(self.dim)

        self.lmbda = kwargs.get('lmbda', 1.0)

        if isinstance(self.lmbda, complex):
            self.numeric_type = np.complex

        self.exact_function = lambda phi_of_time: self.initial_value * np.exp(self.lmbda * phi_of_time)

        self._strings['rhs_wrt_time'] = r"\lambda u(t, \phi(t))"
        self._strings['exact'] = r"e^{\lambda t}"

    def evaluate_wrt_time(self, time, phi_of_time, **kwargs):
        super(LambdaU, self).evaluate_wrt_time(time, phi_of_time, **kwargs)
        if kwargs.get('partial') is not None and isinstance(self.lmbda, complex):
            if isinstance(kwargs['partial'], str) and kwargs['partial'] == 'impl':
                return self.lmbda.real * phi_of_time
            elif kwargs['partial'] == 'expl':
                return self.lmbda.imag * phi_of_time
        else:
            return self.lmbda * phi_of_time

    def direct_implicit(self, *args, **kwargs):
        """Direct Implicit Formula for :math:`u'(t, \\phi_t) &= \\lambda u(t, \\phi_t)`
        """
        assert_named_argument('phis_of_time', kwargs, checking_obj=self)
        assert_named_argument('delta_node', kwargs, checking_obj=self)
        assert_named_argument('integral', kwargs, checking_obj=self)

        _phis = kwargs['phis_of_time']
        assert_is_instance(_phis, list, message="Direct implicit formula needs multiple phis.", checking_obj=self)
        assert_condition(len(_phis) == 3, ValueError, message="Need exactly three different phis.", checking_obj=self)
        for _phi in _phis:
            assert_condition(_phi.shape == self.dim_for_time_solver, ValueError,
                             message="Given phi is of wrong shape: %s != %s" % (_phi.shape, self.dim_for_time_solver),
                             checking_obj=self)

        # _phis[0] : previous iteration -> previous step
        # _phis[1] : previous iteration -> current step
        # _phis[2] : current iteration -> previous step

        _dn = kwargs['delta_node']
        # TODO: make this numerics check more advanced (better warning for critical numerics)
        if isinstance(self.lmbda, complex):
            assert_condition(_dn * self.lmbda.real != 1.0,
                             ArithmeticError, "Direct implicit formula for lambda={:f} and dn={:f} not valid. "
                             .format(self.lmbda, _dn) + "Try implicit solver.",
                             self)
        else:
            assert_condition(_dn * self.lmbda != 1.0,
                             ArithmeticError, "Direct implicit formula for lambda={:f} and dn={:f} not valid. "
                             .format(self.lmbda, _dn) + "Try implicit solver.",
                             self)

        _int = kwargs['integral']

        _fas = kwargs['fas'] \
            if 'fas' in kwargs and kwargs['fas'] is not None else 0.0

        if 'core' in kwargs \
                and (isinstance(kwargs['core'], (ImplicitSdcCore, ImplicitMlSdcCore))
                     or (isinstance(self.lmbda, complex) and isinstance(kwargs['core'], SemiImplicitMlSdcCore))):
            _new = (_phis[2] - _dn * self.lmbda * _phis[1] + _int + _fas) / (1 - self.lmbda * _dn)
            # LOG.debug("Implicit MLSDC Step:\n  %s = (%s - %s * %s * %s + %s + %s) / (1 - %s * %s)"
            #           % (_new, _phis[2], _dn, self.lmbda, _phis[1], _int, _fas, self.lmbda, _dn))
            return _new
        else:
            _new = \
                (_phis[2]
                 + _dn * (complex(0, self.lmbda.imag) * (_phis[2] - _phis[0]) - self.lmbda.real * _phis[1])
                 + _int + _fas) \
                / (1 - self.lmbda.real * _dn)
            # LOG.debug("Semi-Implicit MLSDC Step:\n  %s = (%s + %s * (%s * (%s - %s) - %s * %s) + %s + %s) / (1 - %s * %s)"
            #           % (_new, _phis[2],  _dn, complex(0, self.lmbda.imag), _phis[2], _phis[0], self.lmbda.real, _phis[1], _int, _fas, self.lmbda.real, _dn))
            return _new

    @property
    def lmbda(self):
        return self._lmbda

    @lmbda.setter
    def lmbda(self, lmbda):
        self._lmbda = lmbda

    def print_lines_for_log(self):
        _lines = super(LambdaU, self).print_lines_for_log()
        _lines.update(HasExactSolutionMixin.print_lines_for_log(self))
        _lines['Coefficients'] = '\lambda = %s' % self.lmbda
        return _lines

    def __str__(self):
        _outstr = super(LambdaU, self).__str__()
        _outstr += r", \lambda=%s" % self.lmbda
        _outstr += HasExactSolutionMixin.__str__(self)
        return _outstr
