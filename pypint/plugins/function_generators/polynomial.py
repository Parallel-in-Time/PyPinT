# coding=utf-8
"""

.. moduleauthor:: Dieter Moser <d.moser@fz-juelich.de>
"""
import numpy as np


class PolynomialFG(object):
    """Takes up to multiple :py:class:`numpy.ndarray` and constructs (rational) polynomials.

    Examples
    --------
    .. todo:: add examples for polynomial function generator
    """

    def __init__(self, exponents, coeffs, final_op=None):
        # assert that exponents and coeffs have the right dimensions
        assert isinstance(exponents, np.ndarray) and isinstance(coeffs, np.ndarray)
        if exponents.ndim == 1:
            assert coeffs.ndim == 1 and exponents.size == coeffs.size
            self.dim = 1
        elif exponents.shape[0] == 1 and exponents.ndim == 1:
            exponents = exponents[0, :]
            self.dim = 1
            assert exponents == coeffs.size
        else:
            assert coeffs.ndim == 1 and exponents.ndim == 2 and exponents.shape[1] == coeffs.size
            self.dim = coeffs.shape[0]

        if final_op == None:
            self.f_op = lambda x: x
        elif hasattr(final_op, '__call__'):
            self.f_op = final_op
        else:
            raise ValueError("The final Operation is not callable")

        self.exps = exponents
        self.cs = coeffs

    def generate_function(self):
        if self.exps.ndim > 1:
            def func(x):
                f_x = 0.0
                mult = 1.0
                for i in range(self.cs.size):
                    for j in range(self.exps.shape[0]):
                        mult = mult * (x[j] ** (self.exps[j, i]))
                    f_x = f_x + mult * self.cs[i]
                    mult = 1.0
                return self.f_op(f_x)

        elif self.exps.ndim == 1:
            def func(x):
                f_x = 0.0
                for i in range(self.cs.size):
                    mult = (x ** (self.exps[i]))
                    f_x = f_x + mult * self.cs[i]
                return self.f_op(f_x)
        else:
            func = lambda x: 0.0

        return func
