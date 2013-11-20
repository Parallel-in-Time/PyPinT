# coding=utf-8

from pypint.integrators.weight_function_providers.polynomial_weight_function import PolynomialWeightFunction
import unittest
from nose.tools import *
import numpy as np

test_coefficients = [
    np.asarray([42, 0.0, 3.14])
]

# Tests for w(x) = 1 as weighting function

test_nodes_w_eq_1 = [np.array([-np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0)]),
                     np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)]),
                     np.array([-0.906179845938664, -0.538469310105683, 0.0,
                               0.538469310105683, 0.906179845938664])
]
test_weights_w_eq_1 = [np.array([1.0, 1.0]),
                       np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]),
                       np.array([0.236926885056189, 0.478628670499366,
                                 0.568888888888889, 0.478628670499366,
                                 0.236926885056189])]


def compare_ndarrays(arr1, arr2):
    assert_equal(arr1.size, arr2.size,
                 "Length of the two arrays not equal: {:d} != {:d}"
                 .format(arr1.size, arr2.size))
    for i in range(1, arr1.size):
        assert_almost_equals(arr1.flat[i], arr2.flat[i],
                             msg="{:d}. element not equal:".format(i) +
                                 " arr1[{:d}]={:f} != {:f}=arr2[{:d}]"
                                 .format(i, arr1.flat[i], arr2.flat[i], i),
                             places=None, delta=1e-10)


def init_with_coefficients(coefficients):
    test_obj = PolynomialWeightFunction()
    test_obj.init(coefficients)
    assert_equal(test_obj.coefficients.size, coefficients.size,
                 "Not all coefficients were set.")
    for i in range(0, coefficients.size):
        assert_equal(test_obj.coefficients[i], coefficients[i],
                     "Coefficient {:d} was not set correctly.".format(i))


def add_coefficient(position, value):
    test_obj = PolynomialWeightFunction()
    test_obj.add_coefficient(value, position)
    assert_equal(test_obj.coefficients[position], value)


def init_with_coefficients_tests():
    for coeffs in test_coefficients:
        yield init_with_coefficients, coeffs


def add_coefficient_tests():
    for coeffs in test_coefficients:
        for i in range(0, coeffs.size):
            yield add_coefficient, i, coeffs[i]

#def exact_gauss_legendre_standard_weights(n):

def test_standard_gauss_legendre_weights():
    weights_provider = PolynomialWeightFunction()
    weights_provider.init([1.0])
    for i in range(len(test_nodes_w_eq_1)):
        weights_provider.evaluate(test_nodes_w_eq_1[i],
                                  np.array([-1.0, 1.0]))
        calc_weights = weights_provider.weights
        #print(calc_weights-test_weights_w_eq_1[i])
        yield compare_ndarrays, calc_weights, test_weights_w_eq_1[i]


class PolynomialWeightFunctionTest(unittest.TestCase):
    def setUp(self):
        self._test_obj = PolynomialWeightFunction()

    def test_default_initialization(self):
        self.assertIsNone(self._test_obj.weights,
                          "Weights should be initialized as 'None'")
        self.assertEqual(self._test_obj.coefficients.size, 0,
                         "List of coefficients should be empty")
