# coding=utf-8

from pypint.problems.i_initial_value_problem import IInitialValueProblem
import unittest


class IInitialValueProblemTest(unittest.TestCase):
    def test_initialization(self):
        _test_obj = IInitialValueProblem()
