# coding=utf-8
from pypint.solutions.full_solution import FullSolution
from pypint.solutions.step_solution_data import StepSolutionData
from pypint.solutions.trajectory_solution_data import TrajectorySolutionData
from tests import NumpyAwareTestCase
import numpy
import warnings


class FullSolutionTest(NumpyAwareTestCase):
    def setUp(self):
        self._default = FullSolution()
        self._value1 = numpy.array([1.0, 2.0, 3.0])
        self._value2 = numpy.array([4.0, 5.0, 6.0])
        self._step1 = StepSolutionData(value=self._value1, time_point=1.0)
        self._step2 = StepSolutionData(value=self._value2, time_point=1.5)
        self._step3 = StepSolutionData(value=self._value1, time_point=1.5)
        self._step4 = StepSolutionData(value=self._value1, time_point=2.0)

    def test_adds_solutions_to_data_storage(self):
        _first_iteration = numpy.array([self._step1, self._step2])
        _second_iteration = numpy.array([self._step1, self._step3])
        self._default.add_solution(values=_second_iteration)
        self.assertEqual(self._default.solutions.size, 1)
        self._default.add_solution(iteration=0, values=_first_iteration)
        self.assertEqual(self._default.solutions.size, 2)
        self.assertNumpyArrayEqual(self._default.time_points, numpy.array([1.0, 1.5]))

        self.assertIsInstance(self._default.solution(0), TrajectorySolutionData)

    def test_checks_consistency_of_time_points_on_add(self):
        self._default.add_solution(values=numpy.array([self._step1, self._step3]))
        self._default.add_solution(iteration=0, values=numpy.array([self._step1, self._step2]))
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            self._default.add_solution(values=numpy.array([self._step1, self._step2, self._step4]))
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))

    def test_provides_accessor_for_single_solution(self):
        self.assertIsNone(self._default.solution(-1),
                          "In case no solutions are stored, 'None' should always be returned.")

    def test_provides_array_of_all_solution_data_storages(self):
        self.assertNumpyArrayEqual(self._default.solutions, numpy.array([]))

    def test_provides_time_points(self):
        self.assertIsNone(self._default.time_points)


if __name__ == "__main__":
    import unittest
    unittest.main()
