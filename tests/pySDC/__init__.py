__all__ = []

import unittest
from tests.pySDC.integrate.__init__ import IntegrateTests
from tests.pySDC.sdc_tests import SDCTests

class pySDCTests(unittest.TestSuite):
    def __init__(self):
        self.addTests(IntegrateTests())
        self.addTests(SDCTests())

if __name__ == "__main__":
    unittest.main()
