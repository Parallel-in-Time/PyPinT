# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.multi_level_providers.level_transition_providers.time_transition_provider import TimeTransitionProvider
import numpy
import unittest
from nose.tools import *
from tests import assert_numpy_array_almost_equal


test_data = [
    {
        'coarse_nodes': numpy.linspace(0.0, 1.0, 3),
        'fine_nodes': numpy.linspace(0.0, 1.0, 5),
        'coarse_data': numpy.linspace(1.0, 0.0, 3),
        'fine_data': numpy.linspace(1.0, 0.0, 5)
    }
]


def prolongate(data_set):
    _test_obj = TimeTransitionProvider(coarse_nodes=data_set['coarse_nodes'], fine_nodes=data_set['fine_nodes'])
    prolongated = _test_obj.prolongate(data_set['coarse_data'])
    assert_equal(prolongated.size, data_set['fine_data'].size)
    assert_numpy_array_almost_equal(data_set['fine_data'], prolongated)


def restringate(data_set):
    _test_obj = TimeTransitionProvider(coarse_nodes=data_set['coarse_nodes'], fine_nodes=data_set['fine_nodes'])
    restringated = _test_obj.restringate(data_set['fine_data'])
    assert_equal(restringated.size, data_set['coarse_data'].size)
    assert_numpy_array_almost_equal(data_set['coarse_data'], restringated)


def test_prolongation():
    for data_set in test_data:
        yield prolongate, data_set


def test_restringation():
    for data_set in test_data:
        yield restringate, data_set


class TimeTransitionProviderTest(unittest.TestCase):
    def setUp(self):
        pass


if __name__ == '__main__':
    unittest.main()
