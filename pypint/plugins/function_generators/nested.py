# coding=utf-8
"""

.. moduleauthor:: Dieter Moser <d.moser@fz-juelich.de>
"""


class NestedFG(object):
    """
    Summary
    -------
    Generates nested functions from a function list
    :math:`f_n(f_{n-1}(...f_0(x)...))`.

    Examples
    --------
    .. todo:: add examples for nested function generator
    """

    def __init__(self, functions):
        #assert that each element of functionlist is really a function
        for f in functions:
            assert hasattr(f, '__call__')

        self.funcs = functions

    def generate_function(self):
        def func(x):
            val = self.funcs[0](x)
            for i in range(1, len(self.funcs)):
                val = self.funcs[i](val)
            return val

        return func
