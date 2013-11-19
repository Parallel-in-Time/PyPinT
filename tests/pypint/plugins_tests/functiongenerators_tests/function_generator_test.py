from pypint.plugins.functiongenerators import nested,trigonometric,piecewise,polynomial
import unittest
from nose.tools import *
import numpy as np

# test fields

x_1 = np.arange(10)
x_2 = np.linspace(-1, 1)
x_3 = np.array([np.pi, np.exp(-1), np.sqrt(5)])

test_fields = [x_1,x_2,x_3]

# some functions to compare

def runge_glocke(x):
    return 1/(1+x**2)

def trig_polynom(x):
    return np.sin(5*x)+np.cos(0.5*x)+np.sin(2*x)

def generic_polynom(x):
    return x**5+x*0.5+7*x**3+15

def generic_mdim_polynomial(x):
    return x[0]*x[1]+1
def piecewise_function(x):
    if x < 0:
        return 0.0
    else:
        return 1.0
def nested_function(x):
    return np.exp(np.sin(np.cos(np.sqrt(x**2))))



test_functions = [runge_glocke, trig_polynom, generic_polynom,
                    generic_mdim_polynomial, piecewise_function, nested_function]

# arguments to generate this functions
exponents_runge = np.array([[0,2]])
coeffs_runge = np.array([1.0, 1.0])
func_runge = lambda x : 1/x




test_arguments = {"runge_glocke" :
                        [exponents_runge, coeffs_runge, func_runge],
                  "trig_polynom" :
                        [],


                    }

