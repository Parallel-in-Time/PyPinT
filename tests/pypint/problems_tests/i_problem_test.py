# coding=utf-8

from pypint.problems.i_problem import IProblem
import unittest


class IProblemTest(unittest.TestCase):
    def test_initialization(self):
        _test_obj = IProblem()
