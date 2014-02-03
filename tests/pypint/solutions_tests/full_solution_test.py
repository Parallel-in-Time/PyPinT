# coding=utf-8
import warnings

import numpy

from pypint.solutions.data_storage.step_solution_data import StepSolutionData
from pypint.solutions.data_storage.trajectory_solution_data import TrajectorySolutionData
from pypint.solutions.full_solution import FullSolution
from tests import NumpyAwareTestCase


class FullSolutionTest(NumpyAwareTestCase):
    def setUp(self):
        self._default = FullSolution()
        self._value1 = numpy.array([1.0, 2.0, 3.0])
        self._value2 = numpy.array([4.0, 5.0, 6.0])
        self._step1 = StepSolutionData(value=self._value1, time_point=1.0)
        self._step2 = StepSolutionData(value=self._value2, time_point=1.5)
        self._step3 = StepSolutionData(value=self._value1, time_point=1.5)
        self._step4 = StepSolutionData(value=self._value1, time_point=2.0)
        self._traj1 = TrajectorySolutionData()
        self._traj1.add_solution_data(value=self._value1, time_point=1.0)
        self._traj1.add_solution_data(value=self._value2, time_point=1.5)
        self._traj2 = TrajectorySolutionData()
        self._traj2.add_solution_data(value=self._value1, time_point=1.0)
        self._traj2.add_solution_data(value=self._value1, time_point=1.5)
        self._traj3 = TrajectorySolutionData()
        self._traj3.add_solution_data(value=self._value1, time_point=1.0)
        self._traj3.add_solution_data(value=self._value1, time_point=1.5)
        self._traj3.add_solution_data(value=self._value2, time_point=2.0)

    def test_adds_solutions_to_data_storage(self):
        self._default.add_solution(self._traj1)
        self.assertEqual(len(self._default.solutions), 1)
        self._default.add_solution(self._traj2, iteration=0)
        self.assertEqual(len(self._default.solutions), 2)
        self.assertNumpyArrayEqual(self._default.time_points, numpy.array([1.0, 1.5]))

        self.assertIsInstance(self._default.solution(0), TrajectorySolutionData)

    def test_checks_consistency_of_time_points_on_add(self):
        self._default.add_solution(self._traj1)
        self._default.add_solution(self._traj2, iteration=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            self._default.add_solution(self._traj3)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))

    def test_provides_accessor_for_single_solution(self):
        self.assertIsNone(self._default.solution(-1),
                          "In case no solutions are stored, 'None' should always be returned.")

    def test_provides_array_of_all_solution_data_storages(self):
        self.assertEqual(self._default.solutions, [])

    def test_provides_time_points(self):
        self.assertIsNone(self._default.time_points)


if __name__ == "__main__":
    import unittest
    unittest.main()
