import unittest
from nose.tools import *
import numpy as np
import pySDC.globals as Config
from pySDC.integrate.gauss import Gauss


testNumPoints = [3, 5]
testMethods = ["legendre", "lobatto"]
testCases = {'correct': [], 'fail': []}

testCases['correct'].append({
    'params': {
        'func': lambda t, x: 0.0,
        'begin': 0,
        'end': 1
    },
    'result': 0.0,
    'msg': "Zero function"})
testCases['correct'].append({
    'params': {
        'func': lambda t, x: 1.0,
        'begin': 0,
        'end': 1},
    'result': 1.0,
    'msg': "One function in [0, 1]"})
testCases['correct'].append({
    'params': {
        'func': lambda t, x: 1.0,
        'begin': -3,
        'end': 5},
    'result': 8.0,
    'msg': "One function in [-3, 5]"})
testCases['correct'].append({
    'params': {
        'func': lambda t, x: x ** 2,
        'begin': 0,
        'end': 1},
    'result': 1.0 / 3.0,
    'msg': "x^2 in [0,1]"})
testCases['correct'].append({
    'params': {
        'func': lambda t, x: x,
        'begin': 0,
        'end': 1},
    'result': 0.5,
    'msg': "Identity function in [0, 1]"})
testCases['correct'].append({
    'params': {
        'func': lambda t, x: 1,
        'begin': 0,
        'end': 1},
    'result': 1.0,
    'msg': "One-Function on first two integration nodes in [0,1]"})

testCases['fail'].append({
    'params': {
        'func': lambda t, x: 1.0,
        'begin': 0,
        'end': 0},
    'msg': "Zero interval"})
testCases['fail'].append({
    'params': {
        'func': lambda t, x: 1.0,
        'begin': 1,
        'end': 0},
    'msg': "Negative interval"})

gaussLegendreValues = {
    '2': {'nodes': [-np.sqrt(1.0 / 3.0),
                    np.sqrt(1.0 / 3.0)],
          'weights': [1.0,
                      1.0]},
    '3': {'nodes': [-np.sqrt(3.0 / 5.0),
                    0.0,
                    np.sqrt(3.0 / 5.0)],
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
    assert_equal(len(arr1), len(arr2),
                 "Length of the two arrays not equal: {:d} != {:d}"
                 .format(len(arr1), len(arr2)))
    for i in range(1, len(arr1)):
        assert_almost_equals(arr1[i], arr2[i],
                             msg="{:d}. element not equal:".format(i) +
                                 " arr1[{:d}]={:f} != {:f}=arr2[{:d}]"
                                 .format(i, arr1[i], arr2[i], i),
                             places=None, delta=Config.PRECISION)


def correct_integrate(params, result, message):
    computed = Gauss.integrate(**params)
    assert_almost_equals(computed, result,
                         msg=(message + "\n\tcomputed: {:f}".format(computed) +
                              "\n\texpected: {:f}".format(result)),
                         places=None, delta=Config.PRECISION)


@raises(ValueError)
def failed_integrate(params, message):
    Gauss.integrate(**params)


def compare_computed_legendre_weights(n_points):
    computed = Gauss.legendre_nodes_and_weights(int(n_points))
    compare_arrays(computed['weights'].tolist(),
                   gaussLegendreValues[str(n_points)]['weights'])


def compare_computed_legendre_nodes(n_points):
    computed = Gauss.legendre_nodes_and_weights(int(n_points))
    compare_arrays(computed['nodes'].tolist(),
                   gaussLegendreValues[str(n_points)]['nodes'])


def compare_computed_lobatto_nodes(n_points):
    computed = Gauss.lobatto_nodes(int(n_points))
    expected = Gauss.lobatto_nodes_and_weights(int(n_points))['nodes']
    compare_arrays(computed, expected)


def test_gauss_integrate_correct():
    """
    """
    for method in testMethods:
        for n_points in range(testNumPoints[0], testNumPoints[1] + 1):
            for case in testCases['correct']:
                case['n'] = n_points
                case['method'] = method
                yield correct_integrate, case['params'], \
                    case['result'], case['msg']


def test_gauss_integrate_failures():
    """
    """
    for method in testMethods:
        for n_points in range(testNumPoints[0], testNumPoints[1] + 1):
            for case in testCases['fail']:
                case['n'] = n_points
                case['method'] = method
                yield failed_integrate, case['params'], case['msg']


def test_computed_legendre_nodes():
    """
    """
    for n_points in sorted(gaussLegendreValues.keys()):
        yield compare_computed_legendre_nodes, n_points


def test_computed_legendre_weights():
    """
    """
    for n_points in sorted(gaussLegendreValues.keys()):
        yield compare_computed_legendre_weights, n_points


def test_compute_lobatto_nodes():
    """
    """
    for n_points in range(testNumPoints[0], testNumPoints[1] + 1):
        yield compare_computed_lobatto_nodes, n_points


class GaussTests(unittest.TestCase):
    def test_gauss_initialization(self):
        """
        """
        test_obj = Gauss()
        self.assertIsInstance(test_obj, Gauss)
        self.assertTrue(hasattr(test_obj, 'integrate'),
                        "Newton-Cotes integration scheme needs integrate function.")
        self.assertAlmostEqual(Gauss.integrate(), 1.0, msg="Default integrate values",
                               delta=Config.PRECISION, places=None)

    def test_gauss_integrate_without_points(self):
        """
        """
        with self.assertRaises(ValueError):
            Gauss.integrate(n=0)

    def test_integrate_too_many_points(self):
        """
        """
        with self.assertRaises(NotImplementedError):
            Gauss.integrate(n=10, method='lobatto')


if __name__ == "__main__":
    unittest.main()
