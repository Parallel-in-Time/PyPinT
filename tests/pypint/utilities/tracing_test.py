# coding=utf-8

import unittest
from pypint.utilities.tracing import func_name


class TracingTest(unittest.TestCase):
    def test_func_name(self):
        self.assertRegex(func_name(self), "TracingTest\.test_func_name")
        self.assertRegex(func_name(), "NoneType\.test_func_name")


if __name__ == "__main__":
    unittest.main()
