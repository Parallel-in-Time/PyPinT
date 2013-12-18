# coding=utf-8

import unittest
import numpy
from tests.__init__ import NumpyAwareTestCase
from unittest.mock import patch
from pypint.solvers.sdc import Sdc
from examples.problems.lambda_u import LambdaU
from examples.problems.constant import Constant


class SdcTest(NumpyAwareTestCase):
    def setUp(self):
        self._test_obj = Sdc()

    def test_sdc_solver_initialization(self):
        with patch('pypint.problems.i_initial_value_problem.IInitialValueProblem', spec=True,
                    initial_value=1.0, time_start=0.0, time_end=1.0, numeric_type=numpy.float) as IVP:
            prob = IVP.return_value
            self._test_obj.init(problem=prob)

    def test_constant_minus_one_function(self):
        _constant = Constant(constant=-1.0, shift=1.0)
        self._test_obj.init(problem=_constant)
        _solution = self._test_obj.run()
        self.assertNumpyArrayAlmostEqual(_solution.solution(-1), numpy.array([1.0, 0.5, 0.0]))
        self.assertEqual(_solution.used_iterations, 1,
                         "Explicit SDC should converge in 1 iteration.")

    def test_lambda_u(self):
        _lambda_u = LambdaU(lmbda=-1.0)
        self._test_obj.init(problem=_lambda_u, num_time_steps=2, num_nodes=9)
        _solution = self._test_obj.run()
        self.assertEqual(_solution.used_iterations, 6,
                         "Explicit SDC should converge in 6 iterations.")


if __name__ == "__main__":
    unittest.main()
