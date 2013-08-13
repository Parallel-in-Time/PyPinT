import unittest
from pySDC.integrate.midpoint import *

class MidpointTests( unittest.TestCase ):
    def testInitialization( self ):
        testObj = Midpoint()
        self.assertIsInstance( testObj, Midpoint )
        self.assertTrue( hasattr( testObj, 'integrate' ), "Midpoint integration scheme needs integrate function." )
        Midpoint.integrate()

    def testIntegrate( self ):
        self.assertEqual( Midpoint.integrate(), Decimal( 1.0 ), "Default integrate" )

if __name__ == "__main__":
    unittest.main()
