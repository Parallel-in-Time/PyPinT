# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from pypint.multi_level_providers.multi_level_provider import MultiLevelProvider
import unittest
from unittest.mock import MagicMock
from pypint.integrators.integrator_base import IntegratorBase
from pypint.multi_level_providers.level_transition_providers.i_level_transition_provider \
    import ILevelTransitionProvider
import numpy


class MultiLevelProviderTest(unittest.TestCase):
    def setUp(self):
        # mocks
        self._integrator = MagicMock(name="IntegratorBaseMock")
        self._integrator.__class__ = IntegratorBase
        self._level_transitioner = MagicMock(name="ILevelTransitionProviderMock")
        self._level_transitioner.__class__ = ILevelTransitionProvider

        # test obj
        self._test_obj = MultiLevelProvider(default_transitioner=self._level_transitioner)
        self._test_data = numpy.zeros(4)

    def test_initialization(self):
        _test_obj = MultiLevelProvider()

    def test_add_coarse_level(self):
        self._test_obj.add_coarse_level(self._integrator)

    def test_proxy_integrator(self):
        with self.assertRaises(IndexError):
            self._test_obj.integrator(0)

        # add at least one level
        self._test_obj.add_coarse_level(self._integrator)
        self.assertEqual(self._test_obj.num_levels, 1)
        self.assertIsInstance(self._test_obj.integrator(0), IntegratorBase)

        # try adding a non-IntegratorBase
        with self.assertRaises(ValueError):
            self._test_obj.add_coarse_level(MagicMock(name="NotAnIntegrator"))

    def test_proxy_transitioner(self):
        with self.assertRaises(ValueError):
            self._test_obj.prolongate(coarse_data=self._test_data, coarse_level=0)
        with self.assertRaises(ValueError):
            self._test_obj.restringate(fine_data=self._test_data, fine_level=1)

        # add at two testing levels
        self._test_obj.add_coarse_level(self._integrator)
        self._test_obj.add_coarse_level(self._integrator)
        self.assertEqual(self._test_obj.num_levels, 2)

        #
        self._test_obj.prolongate(coarse_data=self._test_data, coarse_level=1)
        self._level_transitioner.prolongate.assert_called_once_with(self._test_data)
        self._test_obj.restringate(fine_data=self._test_data, fine_level=0)
        self._level_transitioner.restringate.assert_called_once_with(self._test_data)

    def test_special_transitioner(self):
        # add at three testing levels
        self._test_obj.add_coarse_level(self._integrator)
        self._test_obj.add_coarse_level(self._integrator)
        self._test_obj.add_coarse_level(self._integrator)
        self.assertEqual(self._test_obj.num_levels, 3)

        with self.assertRaises(ValueError):
            self._test_obj.add_level_transition(MagicMock(name="NotATransitioner"), coarse_level=2, fine_level=0)

        # mocks
        _special_transitioner = MagicMock(name="SpecialLevelTransitioner")
        _special_transitioner.__class__ = ILevelTransitionProvider

        self._test_obj.add_level_transition(_special_transitioner, coarse_level=2, fine_level=0)
        self._test_obj.prolongate(self._test_data, coarse_level=2, fine_level=0)
        _special_transitioner.prolongate.assert_called_once_with(self._test_data)
        self._test_obj.restringate(self._test_data, coarse_level=2, fine_level=0)
        _special_transitioner.restringate.assert_called_once_with(self._test_data)
