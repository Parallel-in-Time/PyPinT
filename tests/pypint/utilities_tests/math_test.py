# coding=utf-8
import unittest
import numpy
from nose.tools import *

from pypint.utilities.math import *


def lagrange_polynomial_same_point(j, base_points, point):
    assert_equal(lagrange_polynome(j, base_points, point), 1.0)


def lagrange_polynomial_other_point(j, base_points, point):
    assert_equal(lagrange_polynome(j, base_points, point), 0.0)


def test_lagrange_polynomial():
    test_sets = [
        {
            'base_points': numpy.linspace(0.0, 1.0, 3),
        },
        {
            'base_points': numpy.linspace(0.0, 1.0, 5)
        }
    ]
    for _set in test_sets:
        for point in _set['base_points']:
            yield lagrange_polynomial_same_point, _set['base_points'].tolist().index(point), _set['base_points'], point
            for other in _set['base_points']:
                if other != point:
                    yield lagrange_polynomial_other_point, _set['base_points'].tolist().index(point), _set['base_points'], other


class MathTest(unittest.TestCase):
    def setUp(self):
        pass


if __name__ == '__main__':
    unittest.main()
