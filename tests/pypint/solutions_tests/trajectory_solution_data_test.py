# coding=utf-8
from pypint.solutions.trajectory_solution_data import TrajectorySolutionData
from tests import NumpyAwareTestCase
import numpy
import warnings


class TrajectorySolutionDataTest(NumpyAwareTestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=UserWarning)
        self._default = TrajectorySolutionData()
        warnings.resetwarnings()

    def test_provides_values(self):
        self.assertIsNone(self._default.values)
        with self.assertRaises(AttributeError):
            self._default.values = "read-only"

        self.assertRaises(ValueError,
                          TrajectorySolutionData, values="not numpy.ndarray")
        self.assertRaises(ValueError,
                          TrajectorySolutionData, values=numpy.array(["not", "object"]))
        self.assertRaises(ValueError,
                          TrajectorySolutionData, values=numpy.array(["is", "StepSolutionData", False], dtype=object))

    def test_provides_time_points(self):
        self.assertIsNone(self._default.time_points)
        with self.assertRaises(AttributeError):
            self._default.time_points = "read-only"

    def test_provides_errors(self):
        self.assertIsNone(self._default.errors)
        with self.assertRaises(AttributeError):
            self._default.errors = "read-only"

    def test_provides_residuals(self):
        self.assertIsNone(self._default.residuals)
        with self.assertRaises(AttributeError):
            self._default.residuals = "read-only"

if __name__ == '__main__':
    import unittest
    unittest.main()
