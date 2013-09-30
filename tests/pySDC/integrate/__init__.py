__all__ = ["quadrature_tests", "newton_cotes_tests", "gauss_tests"]

import unittest
from tests.pySDC.integrate.quadrature_tests import QuadratureTests
from tests.pySDC.integrate.newton_cotes_tests import *
from tests.pySDC.integrate.gauss_tests import GaussTests


class IntegrateTests(unittest.TestSuite):
    def __init__(self):
        self.addTests(QuadratureTests())
        self.addTests(GaussTests())
        self.addTests(NewtonCotesTests())


if __name__ == "__main__":
    unittest.main()
