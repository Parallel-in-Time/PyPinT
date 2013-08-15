import unittest
from tests.pySDC.integrate.quadrature_tests import QuadratureTests
from tests.pySDC.integrate.newton_cotes_tests import NewtonCotesTests
from tests.pySDC.integrate.gauss_tests import GaussTests

class IntegrateTests( unittest.TestSuite ):
    def __init__( self ):
        self.addTests( QuadratureTests() )
        self.addTests( NewtonCotesTests() )
        self.addTests( GaussTests() )
