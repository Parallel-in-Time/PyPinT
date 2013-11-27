# coding=utf-8

import unittest
import numpy
from tests.__init__ import NumpyAwareTestCase
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from pypint.solvers.sdc import Sdc
from pypint.problems.i_initial_value_problem import IInitialValueProblem
from examples.problems.lambda_u import LambdaU
from examples.problems.constant import Constant


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

    def test_constant_minus_one_function(self):
        _constant = Constant(constant=-1.0, shift=1.0)
        self._test_obj.init(problem=_constant)
        _solution = self._test_obj.run()
        self.assertNumpyArrayAlmostEqual(_solution.solution(-1), numpy.array([1.0, 0.5, 0.0]))
        self.assertEqual(_solution.used_iterations, 1,
                         "Explicit SDC should converge in 1 iteration.")

    def test_lambda_u(self):
        _lambda_u = LambdaU(lmbda=-1.0)
        self._test_obj.init(problem=_lambda_u, num_nodes=9)
        _solution = self._test_obj.run()
        self.assertEqual(_solution.used_iterations, 6,
                         "Explicit SDC should converge in 6 iterations.")


if __name__ == "__main__":
    unittest.main()
