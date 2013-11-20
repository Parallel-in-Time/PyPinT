# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.multi_level_providers.level_transition_providers.full_weighting \
    import FullWeighting
import numpy
import unittest
from nose.tools import *


test_data = [
    {
        "coarse_data": numpy.array([1.0, 3.0, 5.0]),
        "fine_data": numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
    }
]


def prolongate(data_pair):
    full_weighting = FullWeighting(fine_level_points=data_pair["fine_data"].size)
    prolongated = full_weighting.prolongate(data_pair["coarse_data"])
    assert_equal(prolongated.size, data_pair["fine_data"].size)
    # TODO: element-wise compare of prolongated data


def restringate(data_pair):
    full_weighting = FullWeighting(fine_level_points=data_pair["fine_data"].size)
    restringated = full_weighting.restringate(data_pair["fine_data"])
    assert_equal(restringated.size, data_pair["coarse_data"].size)
    # TODO: element-wise compare of restringated data


def test_prolongation():
    for data_pair in test_data:
        yield prolongate, data_pair


def test_restringation():
    for data_pair in test_data:
        yield restringate, data_pair


class FullWeightingTest(unittest.TestCase):
    def test_initialization(self):
        _test_obj = FullWeighting(fine_level_points=5)

    def test_wrong_num_fine_points(self):
        with self.assertRaises(ValueError):
            _test_obj = FullWeighting(fine_level_points=4)
