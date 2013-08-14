import unittest
from decimal import Decimal
from pySDC.integrate.newton_cotes import NewtonCotes

class NewtonCotesTests( unittest.TestCase ):
    def testInitialization( self ):
        testObj = NewtonCotes()
        self.assertIsInstance( testObj, NewtonCotes )
        self.assertTrue( hasattr( testObj, 'integrate' ), "Newton-Cotes integration scheme needs integrate function." )
        NewtonCotes.integrate()

    def testIntegrateOrderOne( self ):
        self.assertEqual( NewtonCotes.integrate(), Decimal( 1.0 ), "Default integrate values" )
        self.assertEqual( NewtonCotes.integrate( lambda x: Decimal( 0.0 ) ), Decimal( 0.0 ), "Zero function" )
        
        testParams = []
        testParams.append( [lambda x: 1.0, 0, 0, 1, 1] )
        testParams.append( [lambda x: 1.0, 0, 1, 0, 1] )
        testParams.append( [lambda x: 1.0, 1, 0, 1, 1] )
        for params in testParams:
            with self.assertRaises( AttributeError ):
                NewtonCotes.integrate( params[0], params[1], params[2], params[3], params[4] )

    def testIntegrateOrderTwo( self ):
        self.assertEqual( NewtonCotes.integrate( lambda x: Decimal( 0.0 ), 0, 1, 10, 2 ), Decimal( 0.0 ), "Zero function" )
        
        testParams = []
        testParams.append( [lambda x: 1.0, 0, 0, 1, 2] )
        testParams.append( [lambda x: 1.0, 0, 1, 0, 2] )
        testParams.append( [lambda x: 1.0, 1, 0, 1, 2] )
        for params in testParams:
            with self.assertRaises( AttributeError ):
                NewtonCotes.integrate( params[0], params[1], params[2], params[3], params[4] )
