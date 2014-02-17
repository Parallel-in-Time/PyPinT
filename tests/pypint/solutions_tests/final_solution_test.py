# coding=utf-8
import warnings

import numpy

from pypint.solutions.final_solution import FinalSolution
from pypint.solvers.diagnosis import Error, Residual
from tests import NumpyAwareTestCase


class FinalSolutionTest(NumpyAwareTestCase):
    def setUp(self):
        self._default = FinalSolution()
        self._value = numpy.array([1.0, 2.0, 3.0])
        self._error = Error(value=self._value)
        self._residual = Residual(value=self._value)

    def test_adds_or_overrides_data_storage(self):
        self._default.add_solution(value=self._value, time_point=1.0)
        self.assertNumpyArrayEqual(self._default.value, self._value)
        self.assertEqual(self._default.time_point, 1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._default.add_solution(value=self._value, time_point=1.0)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))

    def test_proxies_value_of_data_storage(self):
        self.assertIsNone(self._default.value)
        self._default.add_solution(value=self._value, time_point=1.0)
        self.assertNumpyArrayEqual(self._default.value, self._value)

    def test_proxies_time_point_of_data_storage(self):
        self.assertIsNone(self._default.time_point)
        self._default.add_solution(value=self._value, time_point=1.0)
        self.assertNumpyArrayEqual(self._default.value, self._value)

    def test_proxies_error_of_data_storage(self):
        self.assertIsNone(self._default.error)
        self._default.add_solution(value=self._value, error=self._error, time_point=1.0)
        self.assertEqual(self._default.error, self._error)

    def test_proxies_residual_of_data_storage(self):
        self.assertIsNone(self._default.residual)
        self._default.add_solution(value=self._value, residual=self._residual, time_point=1.0)
        self.assertEqual(self._default.residual, self._residual)


if __name__ == "__main__":
    import unittest
    unittest.main()
