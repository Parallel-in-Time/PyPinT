# coding=utf-8
import warnings

import numpy

from pypint.solutions.data_storage.trajectory_solution_data import TrajectorySolutionData
from pypint.solutions.data_storage.step_solution_data import StepSolutionData
from pypint.solvers.diagnosis import Error, Residual
from tests import NumpyAwareTestCase


class TrajectorySolutionDataTest(NumpyAwareTestCase):
    def setUp(self):
        self._default = TrajectorySolutionData()

        self._element1 = StepSolutionData(value=numpy.array([1.0, 2.0]), time_point=0.0,
                                          error=Error(numpy.array([1.0])), residual=Residual(numpy.array([0.5])))
        self._element2 = StepSolutionData(value=numpy.array([1.0, 2.0]), time_point=0.5,
                                          error=Error(numpy.array([1.0])), residual=Residual(numpy.array([0.5])))

    def test_solutions_can_be_added(self):
        self._default.add_solution_data(value=numpy.array([1.0, 2.0]), time_point=0.1)
        self.assertEqual(self._default.data.size, 1)
        self.assertNumpyArrayEqual(self._default.time_points, numpy.array([0.1]))
        self.assertNumpyArrayEqual(self._default.values, numpy.array([[1.0, 2.0]]))

        warnings.simplefilter("ignore")  # each of the following tests emits a warning about failed consistency
        self.assertRaises(ValueError,
                          self._default.add_solution_data, value="not numpy.ndarray", time_point=1.0)
        self.assertRaises(ValueError,
                          self._default.add_solution_data, value=numpy.array(["not", "object"]), time_point=1.0)
        self.assertRaises(ValueError,
                          self._default.add_solution_data,
                          value=numpy.array(["is", "StepSolutionData", False], dtype=object), time_point=1.0)
        self.assertRaises(ValueError,
                          self._default.add_solution_data, value=numpy.array([1.0, 2.0, 3.0]), time_point=1.0)
        warnings.resetwarnings()

    def test_provides_raw_data(self):
        self.assertNumpyArrayEqual(self._default.data, numpy.zeros(0, dtype=numpy.object))
        with self.assertRaises(AttributeError):
            self._default.data = "read-only"

    def test_provides_values(self):
        self.assertNumpyArrayEqual(self._default.values, numpy.zeros(0, dtype=numpy.object))
        with self.assertRaises(AttributeError):
            self._default.values = "read-only"

    def test_provides_time_points(self):
        self.assertNumpyArrayEqual(self._default.time_points, numpy.zeros(0, dtype=numpy.float))
        with self.assertRaises(AttributeError):
            self._default.time_points = "read-only"

    def test_provides_errors(self):
        self.assertNumpyArrayEqual(self._default.errors, numpy.zeros(0, dtype=numpy.object))
        with self.assertRaises(AttributeError):
            self._default.errors = "read-only"

    def test_provides_residuals(self):
        self.assertNumpyArrayEqual(self._default.residuals, numpy.zeros(0, dtype=numpy.object))
        with self.assertRaises(AttributeError):
            self._default.residuals = "read-only"

    def test_is_iterable(self):
        self._default.append(self._element1)
        self.assertTrue(self._element1 in self._default)
        self.assertFalse(self._element2 in self._default)
        self._default.append(self._element2)

        self.assertEqual(len(self._default), 2)
        self.assertEqual(self._default[0], self._element1)
        self.assertEqual(self._default[1], self._element2)

        self._default[1.0] = numpy.array([0.0, 0.5])
        self.assertNumpyArrayEqual(self._default[2].value, numpy.array([0.0, 0.5]))

        for elem in self._default:
            self.assertIsInstance(elem, StepSolutionData)


if __name__ == '__main__':
    import unittest
    unittest.main()
