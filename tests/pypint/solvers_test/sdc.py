# coding=utf-8

import unittest
from pypint.solvers.sdc import Sdc


class SdcTest(unittest.TestCase):
    def test_initialization(self):
        _test_obj = Sdc()
