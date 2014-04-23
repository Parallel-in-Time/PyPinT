# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np

from pypint.problems import IInitialValueProblem, HasExactSolutionMixin, HasDirectImplicitMixin
from pypint.utilities import assert_condition, assert_is_instance, class_name, assert_named_argument
from pypint.solvers.cores.implicit_sdc_core import ImplicitSdcCore
from pypint.solvers.cores.implicit_mlsdc_core import ImplicitMlSdcCore


class LambdaU(IInitialValueProblem, HasExactSolutionMixin, HasDirectImplicitMixin):
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

        if "lmbda" not in kwargs:
            kwargs["lmbda"] = 1.0
        self.lmbda = kwargs["lmbda"]

        if isinstance(self.lmbda, complex):
            self.numeric_type = np.complex

        self.exact_function = lambda phi_of_time: self.initial_value * np.exp(self.lmbda * phi_of_time)

        self._strings["rhs"] = r"\lambda u(t, \phi(t))"
        self._strings["exact"] = r"e^{\lambda t}"

    def evaluate(self, time, phi_of_time, partial=None):
        super(LambdaU, self).evaluate(time, phi_of_time, partial)
        if partial is not None and isinstance(self.lmbda, complex):
            if isinstance(partial, str) and partial == "impl":
                return self.lmbda.imag * phi_of_time
            elif partial == "expl":
                return self.lmbda.real * phi_of_time
        else:
            return self.lmbda * phi_of_time

    def direct_implicit(self, *args, **kwargs):
        """Direct Implicit Formula for :math:`u'(t, \\phi_t) &= \\lambda u(t, \\phi_t)`
        """
        assert_is_instance(self.lmbda, complex,
                           message="Direct implicit formula only valid for imaginay lambda: NOT %s"
                                   % class_name(self.lmbda),
                           checking_obj=self)

        assert_named_argument('phis_of_time', kwargs, checking_obj=self)
        assert_named_argument('delta_node', kwargs, checking_obj=self)
        assert_named_argument('integral', kwargs, checking_obj=self)

        _phis = kwargs["phis_of_time"]
        assert_is_instance(_phis, list, message="Direct implicit formula needs multiple phis.", checking_obj=self)
        assert_condition(len(_phis) == 3, ValueError, message="Need exactly three different phis.", checking_obj=self)

        # _phis[0] : previous iteration -> previous step
        # _phis[1] : previous iteration -> current step
        # _phis[2] : current iteration -> previous step

        _dn = kwargs["delta_node"]
        # TODO: make this numerics check more advanced (better warning for critical numerics)
        assert_condition(_dn * self.lmbda.real != 1.0,
                         ArithmeticError, "Direct implicit formula for lambda={:f} and dn={:f} not valid. "
                                          .format(self.lmbda, _dn) + "Try implicit solver.",
                         self)
        _int = kwargs["integral"]

        if 'core' in kwargs and isinstance(kwargs['core'], (ImplicitSdcCore, ImplicitMlSdcCore)):
            # print("(%s - %s * %s * %s + %s) / (1 - %s * %s)" % (_phis[2], _dn, self.lmbda, _phis[1], _int, self.lmbda, _dn))
            return (_phis[2] - _dn * self.lmbda * _phis[1] + _int) / (1 - self.lmbda * _dn)
        else:
            return \
                (_phis[2]
                 + _dn * (complex(0, self.lmbda.imag) * (_phis[2] - _phis[0]) - self.lmbda.real * _phis[1])
                 + _int) \
                / (1 - self.lmbda.real * _dn)

    @property
    def lmbda(self):
        return self._lmbda

    @lmbda.setter
    def lmbda(self, lmbda):
        self._lmbda = lmbda

    def print_lines_for_log(self):
        _lines = super(LambdaU, self).print_lines_for_log()
        _lines['Coefficients'] = '\lambda = {}'.format(self.lmbda)
        return _lines

    def __str__(self):
        str = super(LambdaU, self).__str__()
        str += r", \lambda={}".format(self.lmbda)
        return str
