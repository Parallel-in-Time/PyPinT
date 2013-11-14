# coding=utf-8

import unittest
from .communicators_tests import *
from .integrators_tests import *
from .multi_level_providers_tests import *
from .plugins_tests import *
from .problems_tests import *
from .solutions_tests import *
from .solvers_test import *

class PyPinTTests(unittest.TestSuite):
    def __init__(self):
        self.addTests(CommunicatorsTests)
        self.addTests(IntegratorsTests)
        self.addTests(MultiLevelProvidersTests)
        self.addTests(PluginsTests)
        self.addTests(ProblemsTests)
        self.addTests(SolutionsTests)
        self.addTests(SolversTests)


if __name__ == "__main__":
    unittest.main()

__all__ = ["PyPinTTests"]
