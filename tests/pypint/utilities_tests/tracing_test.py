# coding=utf-8

import unittest
from pypint.utilities.tracing import func_name


class TracingTest(unittest.TestCase):
    def test_func_name(self):
        self.assertRegex(func_name(self), "^TracingTest<0x[0-9a-f]*>\.test_func_name")
        self.assertRegex(func_name(), "^test_func_name")


if __name__ == "__main__":
    unittest.main()
