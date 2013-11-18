__author__ = 'moser'

class NestedFG:
    """
    Generates nested functions from functionlist
    f_n(f_{n-1}(...f_0(x)...))
    """
    def __init__(self,functions):
        #assert that each element of functionlist is really a function
        for f in functions:
            assert hasattr(f,'__call__')

        self.funcs=functions

    def generate_function(self):
        def func(x):
            val=self.funcs[0](x)
            for i in range(1,len(self.funcs)):
                val=self.funcs[i](val)
            return val

        return func



