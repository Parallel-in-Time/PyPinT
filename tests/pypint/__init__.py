# coding=utf-8

import unittest
import numpy


class PyPinTTests(unittest.TestSuite):
    def __init__(self):
        pass


class NumpyAwareTestCase(unittest.TestCase):
    def assertNumpyArrayAlmostEqual(self, actual, expected, places=None, delta=None):
        self.assertIsInstance(actual, numpy.ndarray, "First argument is not a numpy array")
        self.assertIsInstance(expected, numpy.ndarray, "Second argument is not a numpy array")
        self.assertTupleEqual(actual.shape, expected.shape, "The two arrays have different shape")
        actual_flat = actual.flatten()
        expected_flat = expected.flatten()
        self.assertEqual(actual_flat.size, expected_flat.size, "The two arrays are of different size")

        for index in range(0, actual.size):
            self.assertAlmostEqual(actual_flat[index], expected_flat[index], places=places, delta=delta)


if __name__ == "__main__":
    unittest.main()
