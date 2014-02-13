# coding=utf-8
import unittest
from pypint.utilities.threshold_check import ThresholdCheck


class ThresholdCheckTest(unittest.TestCase):
    def setUp(self):
        self._default = ThresholdCheck()

    def test_has_default_thresholds(self):
        self.assertIs(self._default.max_iterations, 10)
        self.assertEqual(self._default.min_residual, 1e-7)
        self.assertIsNone(self._default.min_error)
        self.assertIsNone(self._default.min_reduction)
        self.assertIsNone(self._default.has_reached())

    def test_prints_conditions(self):
        self.assertRegex(self._default.print_conditions(), "iterations=10")


if __name__ == '__main__':
    unittest.main()
