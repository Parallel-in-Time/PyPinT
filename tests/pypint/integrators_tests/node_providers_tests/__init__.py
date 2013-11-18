# coding=utf-8

import unittest
from .gauss_lobatto_nodes_test import GaussLobattoNodesTest
from .gauss_legendre_nodes_test import GaussLegendreNodesTest


class NodeProvidersTests(unittest.TestSuite):
    def __init__(self):
        self.addTests(GaussLobattoNodesTest)
        self.addTests(GaussLegendreNodesTest)


if __name__ == "__main__":
    unittest.main()
