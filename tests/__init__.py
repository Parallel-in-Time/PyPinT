__all__ = []

import unittest
import nose
from tests.pySDC.__init__ import PySDCTests


class PySDCTestSuite(unittest.TestSuite):
    def __init__(self):
        self.addTests(PySDCTests())


if __name__ == "__main__":
    nose.main()
