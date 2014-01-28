# coding=utf-8
from pypint.solutions.step_solution_data import StepSolutionData
from tests import NumpyAwareTestCase
import numpy
import warnings


class StepSolutionDataTest(NumpyAwareTestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=UserWarning)
        self._default = StepSolutionData()
        warnings.resetwarnings()

        self._value = numpy.array([1.0, 2.0])
        self._error = numpy.array([1.0])
        self._residual = numpy.array([0.01])

    def test_provides_solution_value(self):
        self.assertIsNone(self._default.value, "Default value is 'None'.")
        with self.assertRaises(AttributeError):
            self._default.value = "read-only"

        _test = StepSolutionData(value=self._value.copy())
        self.assertEqual(_test.dim, self._value.size)
        self.assertEqual(_test.numeric_type, self._value.dtype)
        self.assertNumpyArrayEqual(_test.value, self._value)

        with self.assertRaises(ValueError):
            StepSolutionData(value="not numpy.ndarray")

    def test_provides_time_point(self):
        self.assertIsNone(self._default.time_point)
        with self.assertRaises(AttributeError):
            self._default.time_point = "read-only"

        _test = StepSolutionData(time_point=0.0)
        self.assertEqual(_test.time_point, 0.0)

        with self.assertRaises(ValueError):
            StepSolutionData(time_point="not float")

    def test_provides_error(self):
        self.assertIsNone(self._default.error)
        with self.assertRaises(AttributeError):
            self._default.error = "read-only"

        _test = StepSolutionData(error=self._error)
        self.assertNumpyArrayEqual(_test.error, self._error)

        with self.assertRaises(ValueError):
            StepSolutionData(error="not numpy.ndarray")

    def test_provides_residual(self):
        self.assertIsNone(self._default.residual)
        with self.assertRaises(AttributeError):
            self._default.residual = "read-only"

        _test = StepSolutionData(residual=self._residual)
        self.assertNumpyArrayEqual(_test.residual, self._residual)

        with self.assertRaises(ValueError):
            StepSolutionData(residual="not numpy.ndarray")


if __name__ == '__main__':
    import unittest
    unittest.main()
