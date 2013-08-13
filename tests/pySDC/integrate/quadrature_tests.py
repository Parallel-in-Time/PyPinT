import unittest
from pySDC.integrate.quadrature import *

class QuadratureTests( unittest.TestCase ):
    def testInitialization( self ):
        with self.assertRaises( NotImplementedError ):
            Quadrature()
        with self.assertRaises( NotImplementedError ):
            Quadrature.integrate()

if __name__ == "__main__":
    unittest.main()
