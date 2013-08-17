import unittest
from nose.tools import *
from decimal import Decimal, getcontext
from pySDC.integrate.gauss import Gauss

testNumPoints = [ 3, 5 ]
testCases = { 'correct': [], 'fail': [] }

testCases['correct'].append( { 'func': lambda x: Decimal( 0.0 ), 'begin': 0, 'end': 1, 'result': Decimal( 0.0 ), 'msg': "Zero function" } )
testCases['correct'].append( { 'func': lambda x: Decimal( 1.0 ), 'begin': 0, 'end': 1, 'result': Decimal( 1.0 ), 'msg': "One function in [0, 1]" } )
testCases['correct'].append( { 'func': lambda x: Decimal( 1.0 ), 'begin':-3, 'end': 5, 'result': Decimal( 8.0 ), 'msg': "One function in [-3, 5]" } )
testCases['correct'].append( { 'func': lambda x: Decimal( x ), 'begin': 0, 'end': 1, 'result': Decimal( 0.5 ), 'msg': "Identity function in [0, 1]" } )

testCases['fail'].append( { 'func': lambda x: 1.0, 'begin': 0, 'end': 0, 'msg': "Zero interval" } )
testCases['fail'].append( { 'func': lambda x: 1.0, 'begin': 1, 'end': 0, 'msg': "Negative interval" } )

def correct_integrate( func, begin, end, nPoints, result, message ):
    assert_almost_equals( Gauss.integrate( func, begin, end, nPoints ), result, msg=message, places=None, delta=getcontext().prec )

@raises( ValueError )
def failed_integrate( func, begin, end, nPoints, message ):
    Gauss.integrate( func, begin, end, nPoints )

def test_gauss_integrate_correct():
    """
    """
    for nPoints in range( testNumPoints[0], testNumPoints[1] + 1 ):
        for case in testCases['correct']:
            yield correct_integrate, case['func'], case['begin'], case['end'], nPoints, case['result'], case['msg']

def test_gauss_integrate_failures():
    """
    """
    for nPoints in range( testNumPoints[0], testNumPoints[1] + 1 ):
        for case in testCases['fail']:
            yield failed_integrate, case['func'], case['begin'], case['end'], nPoints, case['msg']

class GaussTests( unittest.TestCase ):
    def test_gauss_initialization( self ):
        """
        """
        testObj = Gauss()
        self.assertIsInstance( testObj, Gauss )
        self.assertTrue( hasattr( testObj, 'integrate' ), "Newton-Cotes integration scheme needs integrate function." )
        self.assertAlmostEqual( Gauss.integrate(), Decimal( 1.0 ), msg="Default integrate values", delta=Decimal( 1e-7 ), places=None )

    def test_gauss_integrate_without_points( self ):
        """
        """
        with self.assertRaises( ValueError ):
            Gauss.integrate( nPoints=0 )

    def test_integrate_too_many_points( self ):
        """
        """
        with self.assertRaises( NotImplementedError ):
            Gauss.integrate( nPoints=10 )
