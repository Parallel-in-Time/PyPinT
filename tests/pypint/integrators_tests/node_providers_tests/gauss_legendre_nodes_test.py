# coding=utf-8

from pypint.integrators.node_providers.gauss_legendre_nodes import GaussLegendreNodes
import unittest
from nose.tools import *
import numpy as np


test_num_nodes = range(2, 7)

def manual_initialization(n_nodes):
    nodes = GaussLegendreNodes
    nodes.init(n_nodes)
    assert_equal(nodes.num_nodes, n_nodes,
                 "Number of nodes should be set")
    assert_is_instance(nodes.nodes, np.ndarray,
                       "Nodes should be a numpy.ndarray")
    assert_equal(nodes.nodes.size, n_nodes,
                 "There should be correct number of nodes")


def test_manual_initialization():
    for n_nodes in test_num_nodes:
        yield manual_initialization, n_nodes


class GaussLegendreNodesTest(unittest.TestCase):
    def setUp(self):
        self._test_obj = GaussLegendreNodes()

    def test_default_initialization(self):
        self.assertIsNone(self._test_obj.num_nodes,
                          "Number of nodes should be initialized as 'None'")
        self.assertIsNone(self._test_obj.nodes,
                          "Nodes list should be initializes as 'None'")
