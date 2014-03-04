# coding=utf-8

from nose.tools import *
import unittest
import numpy


class PyPintTestSuite(unittest.TestSuite):
    def __init__(self):
        pass


def assert_numpy_array_almost_equal(first, second, places=None, delta=None):
    assert_is_instance(first, numpy.ndarray,
                       "First argument is not a numpy array: {:s}".format(first.__class__.__name__))
    assert_is_instance(second, numpy.ndarray,
                       "Second argument is not a numpy array: {:s}".format(second.__class__.__name__))
    assert_tuple_equal(first.shape, second.shape,
                       "The two arrays have different shape: {} != {}".format(first.shape, second.shape))
    first_flat = first.flatten()
    second_flat = second.flatten()
    assert_equal(first_flat.size, second_flat.size,
                 "The two arrays are of different size: {:d} != {:d}".format(first_flat.size, second_flat.size))

    for index in range(0, first_flat.size):
        assert_almost_equal(first_flat[index], second_flat[index], places=places, delta=delta,
                            msg="Element {:d} not equal: {:f} != {:f}"
                                .format(index, first_flat[index], second_flat[index]))


def assert_numpy_array_equal(first, second):
    assert_is_instance(first, numpy.ndarray,
                       "First argument is not a numpy array: {:s}".format(first.__class__.__name__))
    assert_is_instance(second, numpy.ndarray,
                       "Second argument is not a numpy array: {:s}".format(second.__class__.__name__))
    assert_tuple_equal(first.shape, second.shape,
                       "The two arrays have different shape: {} != {}".format(first.shape, second.shape))
    first_flat = first.flatten()
    second_flat = second.flatten()
    assert_equal(first_flat.size, second_flat.size,
                 "The two arrays are of different size: {:d} != {:d}".format(first_flat.size, second_flat.size))

    for index in range(0, first_flat.size):
        assert_equal(first_flat[index], second_flat[index],
                     msg="Element {:d} not equal: {:f} != {:f}".format(index, first_flat[index], second_flat[index]))


class NumpyAwareTestCase(unittest.TestCase):
    def assertNumpyArrayEqual(self, first, second):
        self.assertIsInstance(first, numpy.ndarray,
                              "First argument is not a numpy array: {:s}".format(first.__class__.__name__))
        self.assertIsInstance(second, numpy.ndarray,
                              "Second argument is not a numpy array: {:s}".format(second.__class__.__name__))
        self.assertTupleEqual(first.shape, second.shape,
                              "The two arrays have different shape: {} != {}".format(first.shape, second.shape))
        first_flat = first.flatten()
        second_flat = second.flatten()
        self.assertEqual(first_flat.size, second_flat.size,
                         "The two arrays are of different size: {:d} != {:d}".format(first_flat.size, second_flat.size))

        for index in range(0, first.size):
            self.assertEqual(first_flat[index], second_flat[index],
                             msg="Element {:d} not equal: {:f} != {:f}"
                                 .format(index, first_flat[index], second_flat[index]))

    def assertNumpyArrayAlmostEqual(self, first, second, places=None, delta=None):
        self.assertIsInstance(first, numpy.ndarray,
                              "First argument is not a numpy array: {:s}".format(first.__class__.__name__))
        self.assertIsInstance(second, numpy.ndarray,
                              "Second argument is not a numpy array: {:s}".format(second.__class__.__name__))
        self.assertTupleEqual(first.shape, second.shape,
                              "The two arrays have different shape: {} != {}".format(first.shape, second.shape))
        first_flat = first.flatten()
        second_flat = second.flatten()
        self.assertEqual(first_flat.size, second_flat.size,
                         "The two arrays are of different size: {:d} != {:d}".format(first_flat.size, second_flat.size))

        for index in range(0, first.size):
            self.assertAlmostEqual(first_flat[index], second_flat[index],
                                   places=places, delta=delta,
                                   msg="Element {:d} not equal: {:f} != {:f}"
                                       .format(index, first_flat[index], second_flat[index]))


if __name__ == "__main__":
    unittest.main()
