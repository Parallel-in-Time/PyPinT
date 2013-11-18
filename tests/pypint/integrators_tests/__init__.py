# coding=utf-8

import unittest
from .node_providers_tests import NodeProvidersTests
from .weight_function_providers_tests import WeightFunctionProvidersTests
from .integrator_base_test import IntegratorBaseTest


class IntegratorsTests(unittest.TestSuite):
    def __init__(self):
        self.addTests(NodeProvidersTests)
        self.addTests(WeightFunctionProvidersTests)
        self.addTest(IntegratorBaseTest)


if __name__ == "__main__":
    unittest.main()
