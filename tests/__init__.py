__all__ = []

import unittest
import nose
from tests.pySDC.__init__ import pySDCTests

class pySDCTestSuite(unittest.TestSuite):
    def __init__(self):
        self.addTests(pySDCTests())

if __name__ == "__main__":
    nose.main()
