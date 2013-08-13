import unittest
from tests.pySDC.integrate.quadrature_tests import *
from tests.pySDC.integrate.midpoint_tests import *

class IntegrateTests( unittest.TestSuite ):
    def __init__( self ):
        self.addTests( QuadratureTests() )
        self.addTests( MidpointTests() )

if __name__ == "__main__":
    unittest.main()
