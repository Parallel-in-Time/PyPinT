# coding=utf-8
"""

.. moduleauthor:: Dieter Moser <d.moser@fz-juelich.de>
"""

import numpy as np


class TrigonometricFG(object):
    """Takes up to multiple :py:class:`numpy.ndarray` and constructs trigonometric polynomials

    Examples
    --------
    .. todo:: add examples for nested function generator
    """

    def __init__(self, freqs, coeffs, translations=None, final_op=None):
        # assert that frequencies  and coeffs have the right dimensions
        if translations is None:
            translations = np.zeros(coeffs.shape)
            assert isinstance(freqs, np.ndarray) \
                and isinstance(coeffs, np.ndarray)
        else:
            assert isinstance(freqs, np.ndarray) \
                and isinstance(coeffs, np.ndarray) \
                and isinstance(translations, np.ndarray)

        if freqs.ndim == 1:
            assert coeffs.ndim == 1 and freqs.size == coeffs.size == translations.size
            self.dim = 1
        elif freqs.ndim == 2 and freqs.shape[0] == 1:
            freqs = freqs[0, :]
            translations = translations[0, :]
            self.dim = 1
            assert freqs.size == translations.size == coeffs.size
        else:
            assert coeffs.ndim == 1 and freqs.ndim == 2 and freqs.shape[1] == coeffs.size
            self.dim = coeffs.shape[0]

        assert translations.shape == freqs.shape
        if final_op is None:
            self.f_op = lambda x: x
        elif hasattr(final_op, '__call__'):
            self.f_op = final_op
        else:
            raise ValueError("The final Operation is not callable")

        self.freqs = freqs
        self.cs = coeffs
        self.trans = translations

    def generate_function(self):
        if self.freqs.ndim > 1:
            def func(x):
                f_x = 0.0
                mult = 1.0
                for i in range(self.cs.size):
                    for j in range(self.dim):
                        mult = mult * (np.cos(x[j] * (self.freqs[j, i]) + self.trans[j, i]) )
                    f_x = f_x + mult * self.cs[i]
                    mult = 1.0
                return self.f_op(f_x)
        elif self.freqs.ndim == 1:
            def func(x):
                f_x = 0.0
                for i in range(self.cs.size):
                    mult = (np.cos(x * (self.freqs[i]) + self.trans[i]))
                    f_x = f_x + mult * self.cs[i]
                return self.f_op(f_x)
        else:
            func = lambda x: 0.0

        return func
