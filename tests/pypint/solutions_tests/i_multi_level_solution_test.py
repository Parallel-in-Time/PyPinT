# coding=utf-8

import unittest
from pypint.solutions.i_multi_level_solution import IMultiLevelSolution


class IMultiLevelSolutionTest(unittest.TestCase):
    def test_initialization(self):
        _test_obj = IMultiLevelSolution()
