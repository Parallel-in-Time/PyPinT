# coding=utf-8

import unittest
from pypint.solvers.i_parallel_solver import IParallelSolver


class IParallelSolverTest(unittest.TestCase):
    def test_initialization(self):
        _test_obj = IParallelSolver()
