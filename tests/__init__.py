# coding=utf-8

import unittest
from tests.pypint import PyPinTTests


class PyPintTestSuite(unittest.TestSuite):
    def __init__(self):
        self.addTests(PyPinTTests())


if __name__ == "__main__":
    unittest.main()
