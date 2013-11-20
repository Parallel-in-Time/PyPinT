from pypint.plugins.functiongenerators import nested, trigonometric, piecewise, polynomial
import unittest
from nose.tools import *
import numpy as np

# Adding test cases

# Test fields

x_1 = np.arange(10)
x_2 = np.linspace(-1, 1)
x_3 = np.array([np.pi, np.exp(-1), np.sqrt(5)])

test_fields = [x_1, x_2, x_3]
mdim_test_fields = [np.vstack((x_1, x_1))]
# some functions to compare

def runge_glocke(x):
    return 1 / (1 + x ** 2)


def trig_polynom(x):
    return np.cos(5.0 * x) + np.cos(0.5 * x) + 2.0 * np.cos(2.0 * x) + np.sin(x)


def generic_polynom(x):
    return x ** 5 + x * 0.5 + 7 * x ** 3 + 15


def generic_mdim_polynomial(x):
    return x[0] * x[1] + 5 * x[0] + 1


def piecewise_function(x):
    values = np.zeros(x.shape)
    for i in range(x.size):
        if x[i] <= 0:
            values[i] = 0.0
        else:
            values[i] = 1.0
    return values


def nested_function(x):
    return np.cos(np.sqrt(x ** 6))


#test_functions = [runge_glocke, trig_polynom, generic_polynom,
#                    generic_mdim_polynomial, piecewise_function, nested_function]

# Arguments to generate this functions
runge_exp = np.array([0.0, 2.0])
runge_coeffs = np.array([1.0, 1.0])
final_func_runge = lambda x: x ** (-1)

trig_freqs = np.array([[5.0, 0.5, 2.0, 1.0]])
trig_coeffs = np.array([1.0, 1.0, 2.0, 1.0])
#trig_trans = np.array([[(+np.pi/2)/5, 0.0, (+np.pi/2)/2 ]])
trig_trans = np.zeros(trig_freqs.shape)
trig_trans[0, -1] = -np.pi / 2.0

gen_pol_exp = np.array([5.0, 1.0, 3.0, 0.0])
gen_pol_coeffs = np.array([1, 0.5, 7.0, 15.0])

piecewise_functions = [lambda x: 0.0, lambda x: 1.0]
piecewise_points = np.array([0.0])

gen_mdim_pol_exp = np.array([[1, 0, 1], [1, 0, 0]])
gen_mdim_pol_coeffs = np.array([1.0, 1.0, 5.0])

nested_function_list = [lambda x: x ** 6, lambda x: np.sqrt(x), lambda x: np.cos(x)]

test_options = {
    "polynomial": {
        "runge_glocke":
            [[runge_exp, runge_coeffs, final_func_runge], runge_glocke],
        "generic_polynom":
            [[gen_pol_exp, gen_pol_coeffs, None], generic_polynom]},
    "mdim_polynomial": {
        "generic_mdim_polynom":
            [[gen_mdim_pol_exp, gen_mdim_pol_coeffs, None], generic_mdim_polynomial]},
    "trigonometric": {
        "trig_polynom":
            [[trig_freqs, trig_coeffs, trig_trans, None], trig_polynom]},
    "piecewise": {
        "piece_wise_function":
            [[piecewise_functions, piecewise_points], piecewise_function]},
    "nested": {
        "nested_function":
            [[nested_function_list], nested_function]}
}

# End of adding cases

# writing generators for test cases
# compares values of 2 arrays after flattening
def compare_ndarrays(arr1, arr2):
    assert_equal(arr1.size, arr2.size,
                 "Length of the two arrays not equal: {:d} != {:d}"
                 .format(arr1.size, arr2.size))
    for i in range(1, arr1.size):
        assert_almost_equals(arr1.flat[i], arr2.flat[i],
                             msg="{:d}. element not equal:".format(i) +
                                 " arr1[{:d}]={:f} != {:f}=arr2[{:d}]"
                                 .format(i, arr1.flat[i], arr2.flat[i], i),
                             places=None, delta=1e-14)


def correct_polynomial_generated(test_field, test_function, args):
    generator = polynomial.PolynomialFG(args[0], args[1], args[2])
    func = generator.generate_function()
    compare_ndarrays(func(test_field), test_function(test_field))


def correct_mdim_polynomial_generated(test_field, test_function, args):
    generator = polynomial.PolynomialFG(args[0], args[1], args[2])
    func = generator.generate_function()
    compare_ndarrays(func(test_field), test_function(test_field))


def correct_trigonometric_function_generated(test_field, test_function, args):
    generator = trigonometric.TrigonometricFG(args[0], args[1], args[2], args[3])
    func = generator.generate_function()
    compare_ndarrays(func(test_field), test_function(test_field))


def correct_piecewise_function_generated(test_field, test_function, args):
    generator = piecewise.PiecewiseFG(args[0], args[1])
    func = generator.generate_function()
    compare_ndarrays(func(test_field), test_function(test_field))


def correct_nested_function_generated(test_field, test_function, args):
    generator = nested.NestedFG(args[0])
    func = generator.generate_function()
    compare_ndarrays(func(test_field), test_function(test_field))


def test_polynomial_function_generator():
    for test_field in test_fields:
        for cases in test_options["polynomial"]:
            yield correct_polynomial_generated, test_field, test_options["polynomial"][cases][1], \
                  test_options["polynomial"][cases][0]


def test_mdim_polynomial_function_generator():
    for test_field in mdim_test_fields:
        for cases in test_options["mdim_polynomial"]:
            yield correct_polynomial_generated, test_field, test_options["mdim_polynomial"][cases][1], \
                  test_options["mdim_polynomial"][cases][0]


def test_trigonometric_function_generator():
    for test_field in test_fields:
        for cases in test_options["trigonometric"]:
            yield correct_trigonometric_function_generated, test_field, test_options["trigonometric"][cases][1], \
                  test_options["trigonometric"][cases][0]


def test_piecewise_function_generator():
    for test_field in test_fields:
        for cases in test_options["piecewise"]:
            yield correct_piecewise_function_generated, test_field, test_options["piecewise"][cases][1], \
                  test_options["piecewise"][cases][0]


def test_nested_function_generator():
    for test_field in test_fields:
        for cases in test_options["nested"]:
            yield correct_nested_function_generated, test_field, test_options["nested"][cases][1], \
                  test_options["nested"][cases][0]


#class PolynomialFunctionGeneratorTest(unittest.TestCase):
#    def setUp(self):
#        self._test_obj = polynomial.PolynomialFG()
#class TrigonometricFunctionGeneratorTest(unittest.TestCase):
#    def setUp(self):
#        self._test_obj = nested.NestedFG()
#class PiecewiseFunctionGeneratorTest(unittest.TestCase):
#    def setUp(self):
#        self._test_obj = polynomial.PolynomialFG()
#class NestedFunctionGeneratorTest(unittest.TestCase):
#    def setUp(self):
#        self._test_obj = polynomial.PolynomialFG()




# (1) out of range (0<=index<1) in dimension 0


