import unittest
from decimal import Decimal
from pySDC.integrate.newton_cotes import NewtonCotes

class NewtonCotesTests( unittest.TestCase ):
    @classmethod
    def setUpClass( self ):
        self.testFunctions = []
        self.testFunctions.append( { 'func': lambda x: Decimal( 0.0 ), 'begin': 0, 'end': 1, 'steps': 10, 'result': Decimal( 0.0 ), 'msg': "Zero function" } )
        self.testFunctions.append( { 'func': lambda x: Decimal( 1.0 ), 'begin': 0, 'end': 1, 'steps': 10, 'result': Decimal( 1.0 ), 'msg': "One function" } )
        self.testFunctions.append( { 'func': lambda x: Decimal( x ), 'begin': 0, 'end': 1, 'steps': 10, 'result': Decimal( 0.5 ), 'msg': "Identity function" } )

        self.testFailures = []
        self.testFailures.append( { 'func': lambda x: 1.0, 'begin': 0, 'end': 0, 'steps': 1, 'msg': "Zero interval" } )
        self.testFailures.append( { 'func': lambda x: 1.0, 'begin': 0, 'end': 1, 'steps': 0, 'msg': "No steps" } )
        self.testFailures.append( { 'func': lambda x: 1.0, 'begin': 1, 'end': 0, 'steps': 1, 'msg': "Negative interval" } )

    def _functional( self, order ):
        for params in self.testFunctions:
            self.assertEqual( NewtonCotes.integrate( params['func'], params['begin'],
                                                     params['end'], params['steps'], order ),
                              params['result'], msg=params['msg'] )

    def _failures( self, order ):
        for params in self.testFailures:
            with self.assertRaises( ValueError, msg=params['msg'] ):
                NewtonCotes.integrate( params['func'], params['begin'], params['end'], params['steps'], order )


    def testInitialization( self ):
        testObj = NewtonCotes()
        self.assertIsInstance( testObj, NewtonCotes )
        self.assertTrue( hasattr( testObj, 'integrate' ), "Newton-Cotes integration scheme needs integrate function." )
        NewtonCotes.integrate()

    def testIntegrateOrderNone( self ):
        with self.assertRaises( NotImplementedError ):
            NewtonCotes.integrate( order=0 )

    def testIntegrateOrderOne( self ):
        order = 1
        self.assertEqual( NewtonCotes.integrate(), Decimal( 1.0 ), "Default integrate values" )
        self._functional( order )
        self._failures( order )

    def testIntegrateOrderTwo( self ):
        order = 2
        self._functional( order )
        self._failures( order )

    def testIntegrateOrderThree( self ):
        order = 3
        self._functional( order )
        self._failures( order )

    def testIntegrateOrderFour( self ):
        order = 4
        self._functional( order )
        self._failures( order )
