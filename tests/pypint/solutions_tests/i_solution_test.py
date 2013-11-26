# coding=utf-8

import unittest
from pypint.solutions.i_solution import ISolution


class ISolutionTest(unittest.TestCase):
    def test_initialization(self):
        _test_obj = ISolution()
