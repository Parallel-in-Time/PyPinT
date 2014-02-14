# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy
from unittest import TestCase
from nose.tools import *

from tests import NumpyAwareTestCase, assert_numpy_array_equal
from pypint.solvers.states.i_solver_state import IStepState
from pypint.solvers.states.i_solver_state import IStateIterator
from pypint.solvers.states.i_solver_state import IStaticStateIterator
from pypint.solvers.states.i_solver_state import ITimeStepState
from pypint.solvers.states.i_solver_state import IIterationState
from pypint.solvers.states.i_solver_state import ISolverState
from pypint.solutions.data_storage import TrajectorySolutionData
from pypint.solutions import IterativeSolution


class IStepStateTest(TestCase):
    def setUp(self):
        self._default = IStepState()

    def test_on_done_finalizes_solution(self):
        self.assertFalse(self._default.solution.finalized)
        self._default.done()
        self.assertTrue(self._default.solution.finalized)

    def test_proxies_time_point(self):
        self._default.solution.time_point = 1.0
        self.assertIs(self._default.time_point, 1.0)

    def test_provides_delta_tau(self):
        self.assertEqual(self._default.delta_tau, 0.0)

        self._default.delta_tau = 0.1
        self.assertIs(self._default.delta_tau, 0.1)

        with self.assertRaises(ValueError):
            self._default.delta_tau = -0.1


def is_iterable(test_obj, state_class, num):
    assert_is(len(test_obj), num)
    for step in test_obj:
        assert_is_instance(step, state_class)

    for _i in range(1, num):
        test_obj.proceed()
    assert_raises(StopIteration, test_obj.proceed)


def has_current_accessor(test_obj):
    assert_is(test_obj.current_index, 0)
    assert_is(test_obj.current, test_obj._states[0])
    test_obj.proceed()
    assert_is(test_obj.current_index, 1)
    assert_is(test_obj.current, test_obj._states[1])


def has_previous_accessor(test_obj):
    assert_is(test_obj.current_index, 0)
    assert_is_none(test_obj.previous)
    assert_is_none(test_obj.previous_index)
    test_obj.proceed()
    assert_is(test_obj.current_index, 1)
    assert_is(test_obj.previous, test_obj._states[0])
    assert_is(test_obj.previous_index, 0)


def has_next_accessor(test_obj):
    assert_is(test_obj.current_index, 0)
    assert_is(test_obj.next, test_obj._states[1])
    assert_is(test_obj.next_index, 1)

    _num = len(test_obj)
    for _i in range(1, _num):
        test_obj.proceed()

    assert_is(test_obj.current_index, _num - 1)
    assert_is_none(test_obj.next)
    assert_is_none(test_obj.next_index)


def has_first_and_last_accessor(test_obj):
    assert_is(test_obj.first, test_obj[0])
    assert_is(test_obj.first_index, 0)

    assert_is(test_obj.last, test_obj[-1])
    assert_is(test_obj.last_index, len(test_obj) - 1)


def on_finalize_collect_solutions_and_finalize(test_obj):
    _count = 0.0
    for step in test_obj:
        step.solution.value = numpy.array([1.0])
        step.solution.time_point = _count
        _count += 1

    assert_false(test_obj.solution.finalized)
    assert_numpy_array_equal(test_obj.solution.values, numpy.array([]))
    test_obj.finalize()
    assert_true(test_obj.solution.finalized)
    assert_numpy_array_equal(test_obj.solution.values, numpy.array([[1.0], [1.0], [1.0]]))

    # .done should reset the current_index counter
    assert_is(test_obj.current_index, 0)

    assert_raises(RuntimeError, test_obj.finalize)


class IStaticStateIteratorTest(TestCase):
    def setUp(self):
        self._default = IStaticStateIterator(solution_class=TrajectorySolutionData, element_type=IStepState, num_states=3)

    def test_is_iterable(self):
        is_iterable(self._default, IStepState, 3)

    def test_on_proceed_does_not_change_size(self):
        _old_len = len(self._default)
        self._default.proceed()
        self.assertIs(len(self._default), _old_len)

    def test_current_accessor(self):
        has_current_accessor(self._default)

    def test_previous_accessor(self):
        has_previous_accessor(self._default)

    def test_next_accessor(self):
        has_next_accessor(self._default)

    def test_first_and_last_accessor(self):
        has_first_and_last_accessor(self._default)

        _no_states = IStateIterator(solution_class=TrajectorySolutionData, element_type=IStepState)
        assert_is_none(_no_states.first)
        assert_is_none(_no_states.first_index)

        assert_is_none(_no_states.last)
        assert_is_none(_no_states.last_index)

    def test_on_finalize_collect_solutions_and_finalize(self):
        on_finalize_collect_solutions_and_finalize(self._default)


class ITimeStepStateTest(TestCase):
    def setUp(self):
        self._default = ITimeStepState(num_states=3)
        self.assertIs(len(self._default), 3)

    def test_has_same_features_as_state_iterator(self):
        self.setUp()
        is_iterable(self._default, IStepState, 3)
        self.setUp()
        has_current_accessor(self._default)
        self.setUp()
        has_previous_accessor(self._default)
        self.setUp()
        has_next_accessor(self._default)
        self.setUp()
        has_first_and_last_accessor(self._default)
        self.setUp()
        on_finalize_collect_solutions_and_finalize(self._default)

    def test_provides_delta_time_step(self):
        self.assertEqual(self._default.delta_time_step, 0.0)

        self._default.delta_time_step = 1.0
        self.assertIs(self._default.delta_time_step, 1.0)

        with self.assertRaises(ValueError):
            self._default.delta_time_step = -3.4

    def test_has_initial_state(self):
        self.assertIsNotNone(self._default.initial)

    def test_has_proxies_for_step_time_points(self):
        self._default.proceed()
        self.assertIs(self._default.current_time_point, self._default.current_step.time_point)
        self.assertIs(self._default.previous_time_point, self._default.previous_step.time_point)
        self.assertIs(self._default.next_time_point, self._default.next_step.time_point)
        self.assertIs(self._default.time_points.size, 3)

    def test_has_aliases_for_state_accessors(self):
        self.assertIs(self._default.current_step, self._default.current)
        self.assertIs(self._default.current_step_index, self._default.current_index)
        self.assertIs(self._default.next_step, self._default.next)
        self.assertIs(self._default.next_step_index, self._default.next_index)
        self.assertIs(self._default.last_step, self._default.last)
        self.assertIs(self._default.last_step_index, self._default.last_index)

        self.assertIs(self._default.previous_step, self._default.initial)
        self.assertIsNone(self._default.previous_step_index)

        self._default.proceed()
        self.assertIs(self._default.current_step, self._default.current)
        self.assertIs(self._default.current_step_index, self._default.current_index)
        self.assertIs(self._default.previous_step, self._default.previous)
        self.assertIs(self._default.previous_step_index, self._default.previous_index)
        self.assertIs(self._default.next_step, self._default.next)
        self.assertIs(self._default.next_step_index, self._default.next_index)
        self.assertIs(self._default.last_step, self._default.last)
        self.assertIs(self._default.last_step_index, self._default.last_index)


class IIterationStateTest(NumpyAwareTestCase):
    def setUp(self):
        self._default = IIterationState(num_time_steps=3, num_states=4)

    def test_has_same_features_as_state_iterator(self):
        self.setUp()
        is_iterable(self._default, ITimeStepState, 3)
        self.setUp()
        has_current_accessor(self._default)
        self.setUp()
        has_previous_accessor(self._default)
        self.setUp()
        has_next_accessor(self._default)
        self.setUp()
        has_first_and_last_accessor(self._default)

        for _time in self._default:
            is_iterable(_time, IStepState, 4)

        self.setUp()
        for _time in self._default:
            has_current_accessor(_time)

        self.setUp()
        for _time in self._default:
            has_previous_accessor(_time)

        self.setUp()
        for _time in self._default:
            has_next_accessor(_time)

        self.setUp()
        for _time in self._default:
            has_first_and_last_accessor(_time)

    def test_on_finalize_collect_solutions_and_finalize(self):
        _count = 0.0
        for _time in self._default:
            for _step in _time:
                _step.solution.value = numpy.array([1.0])
                _step.solution.time_point = _count
                _count += 1

        self.assertFalse(self._default.solution.finalized)
        self.assertNumpyArrayEqual(self._default.solution.values, numpy.array([]))
        self._default.finalize()
        self.assertTrue(self._default.solution.finalized)
        self.assertNumpyArrayEqual(self._default.solution.values, numpy.array([[1.0]] * 12))

        # .done should reset the current_index counter
        self.assertIs(self._default.current_index, 0)

        self.assertRaises(RuntimeError, self._default.finalize)

    def test_on_proceed_set_initial_step_of_time_step(self):
        self._default.current_time_step.last_step.solution.value = numpy.array([1.0])
        self.assertIsNone(self._default.current_time_step.initial.solution.value)
        self.assertIsNone(self._default.next_time_step.initial.solution.value)
        self.assertNotEqual(self._default.current_time_step.initial, self._default.next_time_step.initial)
        self._default.proceed()
        self.assertIs(self._default.previous_time_step.last_step, self._default.current_time_step.initial)
        self.assertNumpyArrayEqual(self._default.current_time_step.initial.solution.value, numpy.array([1.0]))

    def test_has_proxies_for_steps(self):
        self._default.proceed()
        self.assertIs(self._default.current_step, self._default.current_time_step.current_step)
        self.assertIs(self._default.current_step_index, self._default.current_time_step.current_step_index)
        self.assertIs(self._default.previous_step, self._default.previous_time_step.last)
        self.assertIs(self._default.next_step, self._default.current_time_step.next_step)

        self._default.current_time_step._current_index = self._default.current_time_step.last_step_index
        self.assertIs(self._default.next_step, self._default.next_time_step.first)

        self.assertIs(self._default.final_step, self._default.last_time_step.last)
        self.assertIs(self._default.first_step, self._default.first_time_step.first)

    def test_has_aliases_for_state_accessors(self):
        self.assertIs(self._default.first_time_step, self._default.first)
        self.assertIs(self._default.first_time_step_index, self._default.first_index)
        self.assertIs(self._default.last_time_step, self._default.last)
        self.assertIs(self._default.last_time_step_index, self._default.last_index)

        while self._default.current_index != self._default.last_index:
            self.assertIs(self._default.current_time_step, self._default.current)
            self.assertIs(self._default.current_time_step_index, self._default.current_index)
            self.assertIs(self._default.previous_time_step, self._default.previous)
            self.assertIs(self._default.previous_time_step_index, self._default.previous_index)
            self.assertIs(self._default.next_time_step, self._default.next)
            self.assertIs(self._default.next_time_step_index, self._default.next_index)
            self._default.proceed()


class ISolverStateTest(TestCase):
    def setUp(self):
        self._default = ISolverState(num_nodes=4, num_time_steps=3)
        _initial = IStepState()
        _initial.solution.value = numpy.array([42.21])
        _initial.solution.time_point = 0.1
        self._default.initial = _initial

    def test_initialized_without_states(self):
        self.assertIs(len(self._default), 0)

    def test_has_interval_dimension(self):
        self.assertEqual(self._default.delta_interval, 0.0)
        self._default.delta_interval = 1.0
        self.assertEqual(self._default.delta_interval, 1.0)

        with self.assertRaises(ValueError):
            self._default.delta_interval = -0.1

    def test_proceeds_infinitely(self):
        self.assertIs(len(self._default), 0)
        self._default.proceed()
        self.assertIs(len(self._default), 1)
        self.assertTrue(self._default.is_first_iteration)

    def test_has_accessors_for_number_time_steps_and_nodes(self):
        self.assertIs(self._default.num_nodes, 4)
        self.assertIs(self._default.num_time_steps, 3)

    def test_has_current_accessor(self):
        self.assertIsNone(self._default.current)
        self._default.proceed()
        self.assertIs(self._default.current, self._default[0])

    def test_has_previous_accessor(self):
        self.assertIsNone(self._default.previous)
        self._default.proceed()
        self.assertIsNone(self._default.previous)
        self._default.proceed()
        self.assertIs(self._default.previous, self._default[0])

    def test_has_first_and_last_accessor(self):
        self._default.proceed()
        has_first_and_last_accessor(self._default)

    def test_on_finalize_collect_solutions_and_finalize(self):
        self.assertFalse(self._default.finalized)
        self._default.finalize()
        self.assertTrue(self._default.finalized)

        self.setUp()
        self._default.proceed()
        self._default.finalize()

    def test_has_proxies_for_time_step_and_step(self):
        self._default.proceed()
        self._check_proxies()
        self._default.current_iteration.proceed()
        self._check_proxies()
        self._default.current_time_step.proceed()
        self._check_proxies()

    def test_has_aliases_for_state_accessors(self):
        self._check_iteration_aliases()
        self._default.proceed()
        self._check_iteration_aliases()

    def _check_proxies(self):
        self.assertIs(self._default.current_time_step, self._default.current_iteration.current_time_step)
        self.assertIs(self._default.current_time_step_index, self._default.current_iteration.current_time_step_index)
        self.assertIs(self._default.previous_time_step, self._default.current_iteration.previous_time_step)
        self.assertIs(self._default.next_time_step, self._default.current_iteration.next_time_step)

        self.assertIs(self._default.current_step, self._default.current_iteration.current_time_step.current_step)
        self.assertIs(self._default.current_step_index,
                      self._default.current_iteration.current_time_step.current_step_index)
        self.assertIs(self._default.previous_step, self._default.current_iteration.current_time_step.previous_step)
        self.assertIs(self._default.next_step, self._default.current_iteration.current_time_step.next_step)

    def _check_iteration_aliases(self):
        self.assertIs(self._default.current_iteration, self._default.current)
        self.assertIs(self._default.current_iteration_index, self._default.current_index)
        self.assertIs(self._default.previous_iteration, self._default.previous)
        self.assertIs(self._default.previous_iteration_index, self._default.previous_index)
        self.assertIs(self._default.first_iteration, self._default.first)
        self.assertIs(self._default.last_iteration, self._default.last)
        self.assertIs(self._default.last_iteration_index, self._default.last_index)


if __name__ == "__main__":
    unittest.main()
