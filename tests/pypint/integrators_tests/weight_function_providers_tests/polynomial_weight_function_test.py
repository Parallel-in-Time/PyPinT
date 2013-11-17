# coding=utf-8

from pypint.integrators.weight_function_providers.polynomial_weight_function import PolynomialWeightFunction
import unittest
from nose.tools import *
import numpy as np

test_coefficients = [
    np.asarray([42, 0.0, 3.14])
]


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


class PolynomialWeightFunctionTest(unittest.TestCase):
    def setUp(self):
        self._test_obj = PolynomialWeightFunction()

    def test_default_initialization(self):
        self.assertIsNone(self._test_obj.weights,
                          "Weights should be initialized as 'None'")
        self.assertEqual(self._test_obj.coefficients.size, 0,
                         "List of coefficients should be empty")
