# coding=utf-8
from nose.tools import *
import numpy

from tests import NumpyAwareTestCase, assert_numpy_array_equal
from pypint.integrators.node_providers.i_nodes import INodes


test_data = [
    {
        'from': {
            'interval': [-1.0, 1.0],
            'nodes': [-1.0, 0.0, 1.0]
        },
        'to': {
            'interval': [0.0, 0.5],
            'nodes': [0.0, 0.25, 0.5]
        }
    },
    {
        'from': {
            'interval': [0.0, 0.5],
            'nodes': [0.0, 0.25, 0.5]
        },
        'to': {
            'interval': [0.5, 1.0],
            'nodes': [0.5, 0.75, 1.0]
        }
    }
]


def transform_nodes(from_data, to_data):
    _nodes = INodes()
    _nodes._interval = numpy.array(from_data['interval'])
    _nodes._nodes = numpy.array(from_data['nodes'])
    _nodes.transform(numpy.array(to_data['interval']))
    assert_equal(_nodes.nodes[0], to_data['interval'][0])
    assert_equal(_nodes.nodes[-1], to_data['interval'][1])
    assert_numpy_array_equal(_nodes.nodes, numpy.array(to_data['nodes']))


def test_interval_transformation():
    for data_set in test_data:
        yield transform_nodes, data_set['from'], data_set['to']


class Test(NumpyAwareTestCase):
    def setUp(self):
        self._test_obj = INodes()


if __name__ == '__main__':
    import unittest
    unittest.main()
