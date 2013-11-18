__author__ = 'moser'

import numpy as np

class PolynomialFG:
    """
    takes numpy arrays and
    constructs (rational) polynomials
    """
    def __init__(self,exponents,coeffs,final_op=0):
        # assert that exponents and coeffs have the right dimensions
        assert isinstance(exponents,np.ndarray) and isinstance(coeffs,np.ndarray)
        if exponents.ndim==1:
            assert coeffs.ndim==1 and exponents.size==coeffs.size
            self.dim=1
        else:
            assert coeffs.ndim==1 and exponents.ndim==2 and exponents.shape[1]==coeffs.size
            self.dim=coeffs.shape[0]

        if final_op==0:
            self.f_op=lambda x:x
        elif hasattr(final_op,'__call__'):
            self.f_op=final_op
        else:
            raise ValueError("The final Operation is not callable")

        self.exps = exponents
        self.cs =  coeffs

    def generate_function(self):

        def func(x):
            f_x=0.0
            mult=1.0
            for i in range(self.cs.size):
                for j in range(self.dim):
                    mult=mult*(x[j]**(self.exps[j,i]))
                f_x=f_x+mult*self.cs[i]
            return self.f_op(f_x)

        return func



