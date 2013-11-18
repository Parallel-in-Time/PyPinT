# coding=utf-8

import unittest
from .polynomial_weight_function_test import PolynomialWeightFunctionTest


class WeightFunctionProvidersTests(unittest.TestSuite):
    def __init__(self):
        self.addTest(PolynomialWeightFunctionTest)


if __name__ == "__main__":
    unittest.main()
