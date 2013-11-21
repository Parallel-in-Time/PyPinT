# coding=utf-8

import unittest
import numpy
from tests.__init__ import NumpyAwareTestCase
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from pypint.solvers.sdc import Sdc
from pypint.problems.i_initial_value_problem import IInitialValueProblem


class SdcTest(NumpyAwareTestCase):
    def setUp(self):
        self._test_obj = Sdc()
        self._ivp = MagicMock("Initial Value Problem")
        self._ivp.__class__ = IInitialValueProblem
        self._ivp_iv = PropertyMock("Initial Value Property")
        self._ivp_iv.return_value = 1.0
        type(self._ivp).initial_value = self._ivp_iv
        self._ivp_t0 = PropertyMock("Time Start Property")
        self._ivp_t0.return_value = 0.0
        type(self._ivp).time_start = self._ivp_t0
        self._ivp_t1 = PropertyMock("Time End Property")
        self._ivp_t1.return_value = 1.0
        type(self._ivp).time_end = self._ivp_t1

    def test_solver_initialization(self):
        self._test_obj.init(problem=self._ivp)

    def test_constant_one_function(self):
        self._ivp.evaluate = MagicMock(return_value=1.0)
        self._test_obj.init(self._ivp)
        _solution = self._test_obj.run()
        self.assertNumpyArrayAlmostEqual(_solution.solution(0), numpy.array([0.0, 0.5, 1.0]))


if __name__ == "__main__":
    unittest.main()
