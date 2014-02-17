# coding=utf-8
"""

.. moduleauthor:: Dieter Moser <d.moser@fz-juelich.de>
"""
import numpy as np


class PiecewiseFG(object):
    """Takes a :py:class:`numpy.ndarray` and function list to generate a piecewise function only for
    :math:`R \\rightarrow R` functions, #function = #points + 1.

    Examples
    --------
    .. todo:: add examples for piecewise function generator
    """

    def __init__(self, functions, points):
        # assert points is a numpy array
        assert isinstance(points, np.ndarray) and points.ndim == 1
        # assert right number of points
        assert len(functions) == points.size + 1
        # assert each function in the list is really a function
        for f in functions:
            assert hasattr(f, '__call__')
            # asserts that points are sorted
        for i in range(1, points.size):
            assert points[i - 1] < points[i]

        self.funcs = functions
        self.p = points

    def generate_function(self):
        def func(x):
            val = 0.0
            values = np.zeros(x.shape)
            for j in range(x.size):
                for i in range(self.p.size):
                    if x[j] <= self.p[i] and i == 0:
                        val = self.funcs[0](x)
                    elif x[j] <= self.p[i] and x > self.p[i - 1]:
                        val = self.funcs[i + 1](x)
                    else:
                        val = self.funcs[-1](x)

                values[j] = val

            return values

        return func
