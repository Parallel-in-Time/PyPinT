# coding=utf-8
import numpy as np

from pypint.problems.i_initial_value_problem import IInitialValueProblem
from tests import NumpyAwareTestCase


class IInitialValueProblemTest(NumpyAwareTestCase):
    def setUp(self):
        self._default = IInitialValueProblem()
        self._1d_iv = np.array([[1.0]])
        self._1d_2v_iv = np.array([[1.0, 2.0]])
        self._2d_1v_iv = np.array([[1.0], [2.0]])
        self._2d_2v_iv = np.array([
            [1.0, 2.0], [2.0, 1.0],
            [-1.0, -2.0], [-2.0, -1.0],
            [0.0, 0.0], [0.0, 0.0]
        ])

    def test_provides_initial_value(self):
        self.assertIsNone(self._default.initial_value, "Initially no initial value is given.")

        self._default.initial_value = self._1d_iv
        self.assertEqual(self._default.initial_value, self._1d_iv)

        _test_obj = IInitialValueProblem(initial_value=self._1d_iv)
        self.assertEqual(_test_obj.initial_value, self._1d_iv)

        _test_obj = IInitialValueProblem(dim=(1, 2), initial_value=self._1d_2v_iv)
        self.assertNumpyArrayEqual(_test_obj.initial_value, self._1d_2v_iv)

        _test_obj = IInitialValueProblem(dim=(2, 1), initial_value=self._2d_1v_iv)
        self.assertNumpyArrayEqual(_test_obj.initial_value, self._2d_1v_iv)

        _test_obj = IInitialValueProblem(dim=(3, 2, 2), initial_value=self._2d_2v_iv)
        self.assertNumpyArrayEqual(_test_obj.initial_value, self._2d_2v_iv)

    def test_validates_given_initial_value(self):
        self.assertRaises(ValueError, self._default.initial_value, self._1d_2v_iv)
        self.assertRaises(ValueError, self._default.initial_value, "not a number")

    def test_adds_initial_value_to_debug_string(self):
        self._default.initial_value = self._1d_iv
        self.assertRegex(self._default.__str__(), "u\(0.00\)=\[\[ 1.\]\]")


if __name__ == "__main__":
    import unittest
    unittest.main()
