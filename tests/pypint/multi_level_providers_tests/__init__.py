# coding=utf-8

import unittest
from .level_transition_providers_tests import *

class MultiLevelProvidersTests(unittest.TestSuite):
    def __init__(self):
        self.addTests(LevelTransitionProvidersTests)


if __name__ == "__main__":
    unittest.main()

__all__ = ["MultiLevelProvidersTests"]
