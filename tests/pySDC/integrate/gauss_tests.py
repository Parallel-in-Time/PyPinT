import unittest
from nose.tools import *
import numpy as np
import pySDC.globals as Config
from pySDC.integrate.gauss import Gauss


testNumPoints = [3, 5]
testMethods = ["legendre", "lobatto"]
testCasesFull = {'correct': [], 'fail': []}

testCasesFull['correct'].append({
    'params': {
        'func': lambda t, x: 0.0,
        'begin': 0,
        'end': 1
    },
    'result': 0.0,
    'msg': "Zero function"})
testCasesFull['correct'].append({
    'params': {
        'func': lambda t, x: 1.0,
        'begin': 0,
        'end': 1},
    'result': 1.0,
    'msg': "One function in [0, 1]"})
testCasesFull['correct'].append({
    'params': {
        'func': lambda t, x: 1.0,
        'begin': -3,
        'end': 5},
    'result': 8.0,
    'msg': "One function in [-3, 5]"})
testCasesFull['correct'].append({
    'params': {
        'func': lambda t, x: x ** 2,
        'begin': 0,
        'end': 1},
    'result': 1.0 / 3.0,
    'msg': "x^2 in [0,1]"})
testCasesFull['correct'].append({
    'params': {
        'func': lambda t, x: x,
        'begin': 0,
        'end': 1},
    'result': 0.5,
    'msg': "Identity function in [0, 1]"})

testCasesFull['fail'].append({
    'params': {
        'func': lambda t, x: 1.0,
        'begin': 0,
        'end': 0},
    'msg': "Zero interval"})
testCasesFull['fail'].append({
    'params': {
        'func': lambda t, x: 1.0,
        'begin': 1,
        'end': 0},
    'msg': "Negative interval"})

testValueSetsPartial = [
    {
        'values': {
            '3': [1.0] * 3,
            '4': [1.0] * 4,
            '5': [1.0] * 5
        },
        't_begin': 0.0,
        't_end': 1.0,
        'msg': "Constant One-Function in [0,1]"
    }
]

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


def partial_integrate(values, begin, end, tau, nodes, msg):
    computed = Gauss.partial_integrate(
        values=values,
        t_begin=begin,
        t_end=end,
        tau_end=tau,
        n=len(values),
        method="lobatto")
    _trans = Gauss.transform(begin, end)
    expected = (_trans[0] * nodes[tau + 1] + _trans[1]) - \
               (_trans[0] * nodes[tau] + _trans[1])
    assert_almost_equal(computed, expected,
                        msg=(msg + " on {}. integration node:".format(tau) +
                             "\n\tcomputed: {:f}".format(computed) +
                             "\n\texpected: {:f}".format(expected)),
                        places=None, delta=Config.PRECISION)


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
            for case in testCasesFull['correct']:
                case['n'] = n_points
                case['method'] = method
                yield correct_integrate, case['params'], \
                    case['result'], case['msg']


def test_gauss_integrate_failures():
    """
    """
    for method in testMethods:
        for n_points in range(testNumPoints[0], testNumPoints[1] + 1):
            for case in testCasesFull['fail']:
                case['n'] = n_points
                case['method'] = method
                yield failed_integrate, case['params'], case['msg']


def test_gauss_partial_integrate():
    for test_value_set in testValueSetsPartial:
        for n_points in sorted(test_value_set['values'].keys()):
            n = int(n_points)
            _nodes = Gauss.lobatto_nodes(n)
            for tau in range(0, n-1):
                yield partial_integrate, test_value_set['values'][n_points], \
                    test_value_set['t_begin'], test_value_set['t_end'], tau, \
                    _nodes, test_value_set['msg']

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
