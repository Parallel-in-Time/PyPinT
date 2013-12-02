# coding=utf-8

from tests.__init__ import NumpyAwareTestCase
import unittest
from pypint.solutions.iterative_solution import IterativeSolution
import numpy


class IterativeSolutionTest(NumpyAwareTestCase):
    def setUp(self):
        self._test_obj = IterativeSolution()
        self._test_values = numpy.array([.1, .2, 1])
        self._test_points = numpy.array([1, 2, 3])

    def test_add_solution_with_defaults(self):
        self._test_obj.add_solution(self._test_points, self._test_values)
        self.assertEqual(self._test_obj.used_iterations, 1)
        self.assertNumpyArrayAlmostEqual(self._test_obj.data[0], self._test_values)
        self.assertNumpyArrayAlmostEqual(self._test_obj.data[0], self._test_obj.solution())

    def test_add_solution_with_specific_level(self):
        self._test_obj.add_solution(self._test_points, self._test_values)
        self._test_obj.add_solution(self._test_points, self._test_values, iteration=3)
        self.assertEqual(self._test_obj.used_iterations, 2)
        self.assertEqual(self._test_obj.data.size, 3)
        self.assertNumpyArrayAlmostEqual(self._test_obj.data[0], self._test_values)
        self.assertIsNone(self._test_obj.data[1])
        self.assertNumpyArrayAlmostEqual(self._test_obj.data[2], self._test_values)
        self.assertNumpyArrayAlmostEqual(self._test_obj.data[2], self._test_obj.solution(iteration=3))

    def test_add_solution_not_overriding(self):
        self._test_obj.add_solution(self._test_points, self._test_values, iteration=1)
        with self.assertRaises(ValueError):
            self._test_obj.add_solution(self._test_points, numpy.array([1, 2]), iteration=1)
        self.assertEqual(self._test_obj.used_iterations, 1)
        self.assertNumpyArrayAlmostEqual(self._test_obj.data[0], self._test_values)
        with self.assertRaises(ValueError):
            self._test_obj.solution(iteration=3)
