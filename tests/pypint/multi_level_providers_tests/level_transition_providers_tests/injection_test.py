# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.multi_level_providers.level_transition_providers.injection \
    import Injection
import numpy
import unittest
from nose.tools import *


class InjectionTest(unittest.TestCase):
    def test_initialization(self):
        _test_obj = Injection(fine_level_points=5)

    def test_wrong_num_fine_points(self):
        with self.assertRaises(ValueError):
            _test_obj = Injection(fine_level_points=4)
