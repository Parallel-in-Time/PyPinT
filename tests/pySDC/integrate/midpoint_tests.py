import unittest
from decimal import Decimal
from pySDC.integrate.midpoint import Midpoint

class MidpointTests( unittest.TestCase ):
    def testInitialization( self ):
        testObj = Midpoint()
        self.assertIsInstance( testObj, Midpoint )
        self.assertTrue( hasattr( testObj, 'integrate' ), "Midpoint integration scheme needs integrate function." )
        Midpoint.integrate()

    def testIntegrate( self ):
        self.assertEqual( Midpoint.integrate(), Decimal( 1.0 ), "Default integrate values" )
        self.assertEqual( Midpoint.integrate( lambda x: Decimal( 0.0 ) ), Decimal( 0.0 ), "Zero function" )
        
        testParams = []
        testParams.append( [lambda x: 1.0, 0, 0, 1] )
        testParams.append( [lambda x: 1.0, 0, 1, 0] )
        testParams.append( [lambda x: 1.0, 1, 0, 1] )
        for params in testParams:
            with self.assertRaises( AttributeError ):
                Midpoint.integrate( params[0], params[1], params[2], params[3] )

if __name__ == "__main__":
    unittest.main()
