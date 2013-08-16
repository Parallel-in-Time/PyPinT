import unittest
from decimal import Decimal, getcontext
from pySDC.integrate.gauss import Gauss

class GaussTests( unittest.TestCase ):
    @classmethod
    def setUpClass( self ):
        self.testFunctions = []
        self.testFunctions.append( { 'func': lambda x: Decimal( 0.0 ), 'begin': 0, 'end': 1, 'result': Decimal( 0.0 ), 'msg': "Zero function" } )
        self.testFunctions.append( { 'func': lambda x: Decimal( 1.0 ), 'begin': 0, 'end': 1, 'result': Decimal( 1.0 ), 'msg': "One function in [0, 1]" } )
        self.testFunctions.append( { 'func': lambda x: Decimal( 1.0 ), 'begin':-3, 'end': 5, 'result': Decimal( 8.0 ), 'msg': "One function in [-3, 5]" } )
        self.testFunctions.append( { 'func': lambda x: Decimal( x ), 'begin': 0, 'end': 1, 'result': Decimal( 0.5 ), 'msg': "Identity function in [0, 1]" } )

        self.testFailureCases = []
        self.testFailureCases.append( { 'func': lambda x: 1.0, 'begin': 0, 'end': 0, 'msg': "Zero interval" } )
        self.testFailureCases.append( { 'func': lambda x: 1.0, 'begin': 1, 'end': 0, 'msg': "Negative interval" } )

    def _functional( self, nPoints ):
        for params in self.testFunctions:
            self.assertAlmostEqual( Gauss.integrate( params['func'], params['begin'], params['end'], nPoints ),
                              params['result'], msg=params['msg'], delta=getcontext().prec, places=None )

    def _failures( self, nPoints ):
        for params in self.testFailureCases:
            with self.assertRaises( ValueError, msg=params['msg'] ):
                Gauss.integrate( params['func'], params['begin'], params['end'] )


    def testInitialization( self ):
        testObj = Gauss()
        self.assertIsInstance( testObj, Gauss )
        self.assertTrue( hasattr( testObj, 'integrate' ), "Newton-Cotes integration scheme needs integrate function." )
        self.assertAlmostEqual( Gauss.integrate(), Decimal( 1.0 ), msg="Default integrate values", delta=Decimal( 1e-7 ), places=None )

    def testIntegrateOrderNone( self ):
        with self.assertRaises( ValueError ):
            Gauss.integrate( nPoints=0 )

    def testIntegrateOrderTooHigh( self ):
        with self.assertRaises( NotImplementedError ):
            Gauss.integrate( nPoints=10 )

    def testIntegrateOrderThree( self ):
        order = 3
        self._functional( order )
        self._failures( order )

    def testIntegrateOrderFour( self ):
        nPoints = 4
        self._functional( nPoints )
        self._failures( nPoints )

    def testIntegrateOrderFive( self ):
        nPoints = 5
        self._functional( nPoints )
        self._failures( nPoints )
