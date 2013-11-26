# coding=utf-8
from pypint.plugins.function_generators import polynomial
import unittest
from tests.__init__ import assert_numpy_array_almost_equal
import numpy as np


# Test fields
x_1 = np.arange(10)
x_2 = np.linspace(-1, 1)
x_3 = np.array([np.pi, np.exp(-1), np.sqrt(5)])

test_fields = [x_1, x_2, x_3]
mdim_test_fields = [np.vstack((x_1, x_1))]


# some functions to compare
def runge_glocke(x):
    return 1 / (1 + x ** 2)


def generic_polynom(x):
    return x ** 5 + x * 0.5 + 7 * x ** 3 + 15


def generic_mdim_polynomial(x):
    return x[0] * x[1] + 5 * x[0] + 1


# Arguments to generate this functions
runge_exp = np.array([0.0, 2.0])
runge_coeffs = np.array([1.0, 1.0])
final_func_runge = lambda x: x ** (-1)

gen_pol_exp = np.array([5.0, 1.0, 3.0, 0.0])
gen_pol_coeffs = np.array([1, 0.5, 7.0, 15.0])

gen_mdim_pol_exp = np.array([[1, 0, 1], [1, 0, 0]])
gen_mdim_pol_coeffs = np.array([1.0, 1.0, 5.0])

test_options = {
    "polynomial": {
        "runge_glocke": [
            [runge_exp, runge_coeffs, final_func_runge],
            runge_glocke
        ],
        "generic_polynom": [
            [gen_pol_exp, gen_pol_coeffs, None],
            generic_polynom]
    },
    "mdim_polynomial": {
        "generic_mdim_polynom": [
            [gen_mdim_pol_exp, gen_mdim_pol_coeffs, None],
            generic_mdim_polynomial
        ]
    },
}
# End of adding cases


# writing generators for test cases
def correct_polynomial_generated(test_field, test_function, args):
    generator = polynomial.PolynomialFG(args[0], args[1], args[2])
    func = generator.generate_function()
    assert_numpy_array_almost_equal(func(test_field), test_function(test_field))


def correct_mdim_polynomial_generated(test_field, test_function, args):
    generator = polynomial.PolynomialFG(args[0], args[1], args[2])
    func = generator.generate_function()
    assert_numpy_array_almost_equal(func(test_field), test_function(test_field))


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


class PolynomialFunctionGeneratorTest(unittest.TestCase):
    pass
