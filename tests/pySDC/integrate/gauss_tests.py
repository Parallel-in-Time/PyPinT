import unittest
from nose.tools import *
import numpy as np
import pySDC.settings as config
from pySDC.integrate.gauss import Gauss


testNumPoints = [ 3, 5 ]
testMethods = ["legendre", "lobatto"]
testCases = { 'correct': [], 'fail': [] }

testCases['correct'].append({ 'func': lambda t, x: 0.0, 'begin': 0, 'end': 1, 'result': 0.0, 'msg': "Zero function" })
testCases['correct'].append({ 'func': lambda t, x: 1.0, 'begin': 0, 'end': 1, 'result': 1.0, 'msg': "One function in [0, 1]" })
testCases['correct'].append({ 'func': lambda t, x: 1.0, 'begin':-3, 'end': 5, 'result': 8.0, 'msg': "One function in [-3, 5]" })
testCases['correct'].append({ 'func': lambda t, x: x, 'begin': 0, 'end': 1, 'result': 0.5, 'msg': "Identity function in [0, 1]" })

testCases['fail'].append({ 'func': lambda t, x: 1.0, 'begin': 0, 'end': 0, 'msg': "Zero interval" })
testCases['fail'].append({ 'func': lambda t, x: 1.0, 'begin': 1, 'end': 0, 'msg': "Negative interval" })

gaussLegendreValues = {
    '2': {'nodes': [-np.sqrt(1.0 / 3.0),
                    np.sqrt(1.0 / 3.0)],
          'weights': [1.0,
                      1.0]},
    '3': {'nodes': [-np.sqrt(3.0 / 5.0),
                    0.0,
                    np.sqrt(3.0 / 5.0) ],
          'weights': [5.0 / 9.0,
                      8.0 / 9.0,
                      5.0 / 9.0]},
    '4': {'nodes': [-np.sqrt(3.0 / 7.0 + 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
                    - np.sqrt(3.0 / 7.0 - 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
                    np.sqrt(3.0 / 7.0 - 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
                    np.sqrt(3.0 / 7.0 + 2.0 / 7.0 * np.sqrt(6.0 / 5.0))],
          'weights': [(18.0 - np.sqrt(30.0)) / 36.0,
                      (18.0 + np.sqrt(30.0)) / 36.0,
                      (18.0 + np.sqrt(30.0)) / 36.0,
                      (18.0 - np.sqrt(30.0)) / 36.0]},
    '5': {'nodes': [-1.0 / 3.0 * np.sqrt(5.0 + 2 * np.sqrt(10.0 / 7.0)),
                    - 1.0 / 3.0 * np.sqrt(5.0 - 2 * np.sqrt(10.0 / 7.0)),
                    0.0,
                    1.0 / 3.0 * np.sqrt(5.0 - 2 * np.sqrt(10.0 / 7.0)),
                    1.0 / 3.0 * np.sqrt(5.0 + 2 * np.sqrt(10.0 / 7.0))],
          'weights': [(322.0 - 13.0 * np.sqrt(70.0)) / 900.0,
                      (322.0 + 13.0 * np.sqrt(70.0)) / 900.0,
                      128.0 / 225.0,
                      (322.0 + 13.0 * np.sqrt(70.0)) / 900.0,
                      (322.0 - 13.0 * np.sqrt(70.0)) / 900.0]}
}

def compare_arrays(arr1, arr2):
#     print("\narr1: " + str(np.around(arr1, config.PRECISION)) + "\narr2: " + str(np.around(arr2, config.PRECISION)))
    assert_equal(len(arr1), len(arr2), "Length of the two arrays not equal: " + str(len(arr1)) + " != " + str(len(arr2)))
    for i in range(1, len(arr1)):
        assert_almost_equals(arr1[i], arr2[i],
                             msg=str(i) + ". element not equal: arr1[" + str(i) + "]=" + str(arr1[i]) + " != " + str(arr2[i]) + "=arr2[" + str(i) + "]", 
                             places=None, delta=config.PRECISION)

def correct_integrate(func, begin, end, nPoints, method, result, message):
    computed = Gauss.integrate(func, begin, end, nPoints, type=method)
    assert_almost_equals(computed, result,
                         msg=message + "\n\tcomputed: " + str(computed) + "\n\texpected: " + str(result),
                         places=None, delta=config.PRECISION)

@raises(ValueError)
def failed_integrate(func, begin, end, nPoints, method, message):
    Gauss.integrate(func, begin, end, nPoints, type=method)

def compare_computed_legendre_weights(nPoints):
    computed = Gauss.legendre_nodes_and_weights(nPoints)
    compare_arrays(computed['weights'].tolist(), gaussLegendreValues[str(nPoints)]['weights'])

def compare_computed_legendre_nodes(nPoints):
    computed = Gauss.legendre_nodes_and_weights(nPoints)
    compare_arrays(computed['nodes'].tolist(), gaussLegendreValues[str(nPoints)]['nodes'])

def test_gauss_integrate_correct():
    """
    """
    for type in testMethods:
        for nPoints in range(testNumPoints[0], testNumPoints[1] + 1):
            for case in testCases['correct']:
                yield correct_integrate, case['func'], case['begin'], case['end'], nPoints, type, case['result'], case['msg']

def test_gauss_integrate_failures():
    """
    """
    for type in testMethods:
        for nPoints in range(testNumPoints[0], testNumPoints[1] + 1):
            for case in testCases['fail']:
                yield failed_integrate, case['func'], case['begin'], case['end'], nPoints, type, case['msg']

def test_computed_legendre_nodes():
    """
    """
    for nPoints in sorted(gaussLegendreValues.keys()):
        yield compare_computed_legendre_nodes, nPoints

def test_computed_legendre_weights():
    """
    """
    for nPoints in sorted(gaussLegendreValues.keys()):
        yield compare_computed_legendre_weights, nPoints

class GaussTests(unittest.TestCase):
    def test_gauss_initialization(self):
        """
        """
        testObj = Gauss()
        self.assertIsInstance(testObj, Gauss)
        self.assertTrue(hasattr(testObj, 'integrate'), "Newton-Cotes integration scheme needs integrate function.")
        self.assertAlmostEqual(Gauss.integrate(), 1.0, msg="Default integrate values", delta=config.PRECISION, places=None)

    def test_gauss_integrate_without_points(self):
        """
        """
        with self.assertRaises(ValueError):
            Gauss.integrate(nPoints=0)

    def test_integrate_too_many_points(self):
        """
        """
        with self.assertRaises(NotImplementedError):
            Gauss.integrate(nPoints=10, type='lobatto')

if __name__ == "__main__":
    unittest.main()
