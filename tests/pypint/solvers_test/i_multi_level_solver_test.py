# coding=utf-8

import unittest
from pypint.solvers.i_multi_level_solver import IMultiLevelSolver


class IMultiLevelSolverTest(unittest.TestCase):
    def test_initialization(self):
        _test_obj = IMultiLevelSolver()
