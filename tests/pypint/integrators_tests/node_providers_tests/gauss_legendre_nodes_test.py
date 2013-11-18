# coding=utf-8

from pypint.integrators.node_providers.gauss_legendre_nodes import GaussLegendreNodes
import unittest
from nose.tools import *
import numpy as np


test_num_nodes = range(2, 7)

def manual_initialization(n_nodes):
    nodes = GaussLegendreNodes()
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

    def test_correctness_of_selected_nodes(self):
        self._test_obj.init(1)
        self.assertAlmostEqual(self._test_obj.nodes[0], 0.0)

        self.setUp()
        self._test_obj.init(2)
        self.assertAlmostEqual(self._test_obj.nodes[0], -np.sqrt(1.0 / 3.0))
        self.assertAlmostEqual(self._test_obj.nodes[1], np.sqrt(1.0 / 3.0))

        self.setUp()
        self._test_obj.init(5)
        self.assertAlmostEqual(self._test_obj.nodes[0], -1.0 / 3.0 * np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)))
        self.assertAlmostEqual(self._test_obj.nodes[1], -1.0 / 3.0 * np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)))
        self.assertAlmostEqual(self._test_obj.nodes[2], 0.0)
        self.assertAlmostEqual(self._test_obj.nodes[3], 1.0 / 3.0 * np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)))
        self.assertAlmostEqual(self._test_obj.nodes[4], 1.0 / 3.0 * np.sqrt(5 + 2 * np.sqrt(10.0 / 7.0)))
