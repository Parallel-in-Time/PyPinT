# coding=utf-8

import unittest
import nose
from tests.pypint import *


class PyPintTestSuite(unittest.TestSuite):
    def __init__(self):
        self.addTests(PyPinTTests())


if __name__ == "__main__":
    nose.main()

__all__ = ["PyPintTestSuite"]
