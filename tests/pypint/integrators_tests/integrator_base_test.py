# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.integrators.integrator_base import IntegratorBase
from pypint.integrators import INTEGRATOR_PRESETS
import numpy
import unittest
from nose.tools import *


def init_with_presets(preset):
    integrator = IntegratorBase()
    integrator.init(**preset)
    assert_is_instance(integrator.nodes, numpy.ndarray)
    assert_equal(integrator.nodes.size, preset["num_nodes"])
    assert_is_instance(integrator.weights, numpy.ndarray)
    assert_equal(integrator.weights.size, preset["num_nodes"])


def test_init_with_presets():
    for preset in INTEGRATOR_PRESETS:
        yield init_with_presets, INTEGRATOR_PRESETS[preset]


class IntegratorBaseTest(unittest.TestCase):
    def test_initialization(self):
        integrator = IntegratorBase()
