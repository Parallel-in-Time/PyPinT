# coding=utf-8

import unittest
from .node_providers_tests import *
from .weight_function_providers_tests import *

class IntegratorsTests(unittest.TestSuite):
    def __init__(self):
        self.addTests(NodeProvidersTests)
        self.addTests(WeightFunctionProvidersTests)


if __name__ == "__main__":
    unittest.main()

__all__ = ["IntegratorsTests"]
