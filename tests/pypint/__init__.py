# coding=utf-8

import unittest
from .communicators_tests import CommunicatorsTests
from .integrators_tests import IntegratorsTests
from .multi_level_providers_tests import MultiLevelProvidersTests
from .plugins_tests import PluginsTests
from .problems_tests import ProblemsTests
from .solutions_tests import SolutionsTests
from .solvers_test import SolversTests


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
