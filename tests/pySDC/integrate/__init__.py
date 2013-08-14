import unittest
from tests.pySDC.integrate.quadrature_tests import QuadratureTests
from tests.pySDC.integrate.newton_cotes_tests import NewtonCotesTests

class IntegrateTests( unittest.TestSuite ):
    def __init__( self ):
        self.addTests( QuadratureTests() )
        self.addTests( NewtonCotesTests() )

if __name__ == "__main__":
    unittest.main()
