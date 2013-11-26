# coding=utf-8

import unittest
from pypint.solvers.i_iterative_time_solver import IIterativeTimeSolver


class IIterativeTimeSolverTest(unittest.TestCase):
    def test_initialization(self):
        _test_obj = IIterativeTimeSolver()
