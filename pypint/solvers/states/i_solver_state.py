# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.kaltt@fz-juelich.de>
"""
from copy import deepcopy

import numpy as np

from pypint.solutions.data_storage import StepSolutionData, TrajectorySolutionData
from pypint.solutions import IterativeSolution
from pypint.utilities import assert_is_key, assert_condition


class IStepState(object):
    def __init__(self, **kwargs):
        self._solution = StepSolutionData()
        self._delta_tau = 0.0

    def done(self):
        self.solution.finalize()

    @property
    def solution(self):
        """
        Returns
        -------
        solution : :py:class:`.StepSolutionData`
        """
        return self._solution

    @property
    def time_point(self):
        """
        Summary
        -------
        Proxy for :py:attr:`.StepSolutionData.time_point`.
        """
        return self._solution.time_point

    @property
    def delta_tau(self):
        return self._delta_tau

    @delta_tau.setter
    def delta_tau(self, delta_tau):
        assert_condition(delta_tau > 0.0,
                         ValueError, "Delta tau must be non-zero positive: NOT {}".format(delta_tau),
                         self)
        self._delta_tau = delta_tau

    def __str__(self):
        return "{}(solution={})".format(self.__class__.__name__, self.solution)


class IStateIterator(object):
    def __init__(self, **kwargs):
        assert_is_key(kwargs, 'solution_class', "Solution type must be given.")
        assert_is_key(kwargs, 'element_type', "Element type must be given.")
        self._solution = kwargs['solution_class']()
        del kwargs['solution_class']
        self._element_type = kwargs['element_type']
        del kwargs['element_type']
        self._states = []
        self._current_index = 0
        self._finalized = False

        if 'num_states' in kwargs:
            _num_states = kwargs['num_states']
            assert_condition(isinstance(_num_states, int) and _num_states > 0,
                             ValueError, "Number of states must be a non-zero positive integer: NOT {}"
                                         .format(_num_states),
                             self)
            self._states = [self._element_type(**kwargs) for i in range(0, _num_states)]

    def finalize(self):
        assert_condition(not self.finalized,
                         RuntimeError, "This {} is already done.".format(self.__class__.__name__),
                         self)
        for _state in self:
            self.solution.add_solution_data(deepcopy(_state.solution))
        self.solution.finalize()
        self._current_index = 0
        self._finalized = True

    @property
    def finalized(self):
        if self._finalized:
            # if this throws, something is really broken
            assert_condition(self.solution.finalized,
                             RuntimeError, "State is finalized but not its solution object.",
                             self)
        return self._finalized

    @property
    def solution(self):
        return self._solution

    @property
    def current(self):
        return self[self.current_index] if self.current_index is not None else None

    @property
    def current_index(self):
        return self._current_index if len(self) > self._current_index else None

    @property
    def previous(self):
        return self[self.previous_index] if self.previous_index is not None else None

    @property
    def previous_index(self):
        return self.current_index - 1 if self.current_index is not None and self.current_index  > 0 else None

    @property
    def first(self):
        return self[self.first_index] if self.first_index is not None else None

    @property
    def first_index(self):
        return 0 if len(self) > 0 else None

    @property
    def last(self):
        return self[self.last_index] if self.last_index is not None else None

    @property
    def last_index(self):
        return len(self) - 1 if len(self) > 0 else None

    def __len__(self):
        return len(self._states)

    def __iter__(self):
        return iter(self._states)

    def __getitem__(self, item):
        return self._states[item]

    def __str__(self):
        _states = [state.__str__() for state in self._states]
        return "{}({}, solution={}, _states={})".format(self.__class__.__name__, self._element_type.__name__,
                                                        self.solution.__str__(), _states.__str__())


class IStaticStateIterator(IStateIterator):
    def proceed(self):
        assert_condition(not self.finalized,
                         RuntimeError, "This {} is already done.".format(self.__class__.__name__),
                         self)
        if self.next_index is not None:
            self._current_index += 1
        else:
            raise StopIteration("No further states available.")

    @property
    def next(self):
        return self[self.next_index] if self.next_index is not None else None

    @property
    def next_index(self):
        return self.current_index + 1 if self.current_index is not None and len(self) > self.current_index + 1 else None


class ITimeStepState(IStaticStateIterator):
    """
    Summary
    -------
    Stores integration step states of a single time step.
    """
    def __init__(self, **kwargs):
        if 'solution_class' not in kwargs:
            kwargs['solution_class'] = TrajectorySolutionData
        if 'element_type' not in kwargs:
            kwargs['element_type'] = IStepState
        assert_is_key(kwargs, 'num_states', "Number of states must be given.")
        super(ITimeStepState, self).__init__(**kwargs)

        self._delta_time_step = 0.0
        self._initial = IStepState()

    @property
    def delta_time_step(self):
        return self._delta_time_step

    @delta_time_step.setter
    def delta_time_step(self, delta_time_step):
        assert_condition(delta_time_step > 0.0,
                         ValueError, "Delta interval must be non-zero positive: NOT {}".format(delta_time_step),
                         self)
        self._delta_time_step = delta_time_step

    @property
    def initial(self):
        return self._initial

    @initial.setter
    def initial(self, initial):
        self._initial = initial

    @property
    def time_points(self):
        return np.array([step.time_point for step in self], dtype=float)

    @property
    def current_time_point(self):
        return self.current_step.time_point if self.current_step else None

    @property
    def previous_time_point(self):
        return self.previous_step.time_point if self.previous else None

    @property
    def next_time_point(self):
        return self.next.time_point if self.next else None

    @property
    def current_step(self):
        return self.current

    @property
    def current_step_index(self):
        return self.current_index

    @property
    def previous_step(self):
        return self.previous if self.previous is not None else self.initial

    @property
    def previous_step_index(self):
        return self.previous_index

    @property
    def next_step(self):
        return self.next

    @property
    def next_step_index(self):
        return self.next_index

    @property
    def last_step(self):
        return self.last

    @property
    def last_step_index(self):
        return self.last_index


class IIterationState(IStaticStateIterator):
    """
    Summary
    -------
    Stores time step states of a single iteration.
    """
    def __init__(self, **kwargs):
        if 'solution_class' not in kwargs:
            kwargs['solution_class'] = TrajectorySolutionData
        if 'element_type' not in kwargs:
            kwargs['element_type'] = ITimeStepState
        super(IIterationState, self).__init__(**kwargs)
        del kwargs['solution_class']
        del kwargs['element_type']

        assert_is_key(kwargs, 'num_time_steps', "Number of time steps must be given.")
        _num_time_steps = kwargs['num_time_steps']
        del kwargs['num_time_steps']
        self._states = [self._element_type(**kwargs) for i in range(0, _num_time_steps)]

        self._delta_interval = 0.0
        self._initial_state = self.first_time_step.initial

    def finalize(self):
        assert_condition(not self.finalized,
                         RuntimeError, "This {} is already done.".format(self.__class__.__name__),
                         self)
        for _time_step in self:
            for _step in _time_step:
                self.solution.add_solution_data(deepcopy(_step.solution))
        self.solution.finalize()
        self._current_index = 0
        self._finalized = True

    def proceed(self):
        super(IIterationState, self).proceed()  # -> current_index += 1
        # link initial step of this time step to the previous' last step
        self.current_time_step.initial = self.previous_time_step.last_step

    @property
    def current_time_step(self):
        return self.current

    @property
    def current_time_step_index(self):
        return self.current_index

    @property
    def previous_time_step(self):
        return self.previous

    @property
    def previous_time_step_index(self):
        return self.previous_index

    @property
    def next_time_step(self):
        return self.next

    @property
    def next_time_step_index(self):
        return self.next_index

    @property
    def first_time_step(self):
        return self.first

    @property
    def first_time_step_index(self):
        return self.first_index

    @property
    def last_time_step(self):
        return self.last

    @property
    def last_time_step_index(self):
        return self.last_index

    @property
    def current_step(self):
        return self.current_time_step.current_step

    @property
    def current_step_index(self):
        return self.current_time_step.current_step_index

    @property
    def previous_step(self):
        return self.current_time_step.previous_step \
            if self.current_time_step.previous_step else self.previous_time_step.last

    @property
    def next_step(self):
        return self.current_time_step.next_step if self.current_time_step.next_step else self.next_time_step.first

    @property
    def first_step(self):
        return self.first_time_step.first if self.first_time_step else None

    @property
    def final_step(self):
        return self.last_time_step.last if self.last_time_step else None

    @property
    def time_points(self):
        return np.array([_step.time_point for _step in [_time for _time in self]], dtype=float)


class ISolverState(IStateIterator):
    """
    Summary
    -------
    Stores iteration states.
    """
    def __init__(self, **kwargs):
        if 'solution_class' not in kwargs:
            kwargs['solution_class'] = IterativeSolution
        if 'element_type' not in kwargs:
            kwargs['element_type'] = IIterationState
        super(ISolverState, self).__init__(**kwargs)

        self._num_nodes = kwargs['num_nodes'] if 'num_nodes' in kwargs else 0
        self._num_time_steps = kwargs['num_time_steps'] if 'num_time_steps' in kwargs else 0
        self._delta_interval = 0.0
        self._initial = IStepState()

    def proceed(self):
        self._add_iteration()
        self._current_index = len(self) - 1
        self.current_iteration.initial = self.initial
        self.current_time_step.initial = self.current_iteration.initial

    def finalize(self):
        assert_condition(not self.finalized,
                         RuntimeError, "This {} is already done.".format(self.__class__.__name__),
                         self)
        for _iter in self:
            self.solution.add_solution(_iter.solution)
        self.solution.finalize()
        self._current_index = 0
        self._finalized = True

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_time_steps(self):
        return self._num_time_steps

    @property
    def delta_interval(self):
        return self._delta_interval

    @delta_interval.setter
    def delta_interval(self, delta_interval):
        assert_condition(delta_interval > 0.0,
                         ValueError, "Delta interval must be non-zero positive: NOT {}".format(delta_interval),
                         self)
        self._delta_interval = delta_interval

    @property
    def initial(self):
        return self._initial

    @initial.setter
    def initial(self, initial):
        self._initial = initial

    @property
    def current_iteration(self):
        return self.current

    @property
    def current_iteration_index(self):
        return self.current_index

    @property
    def previous_iteration(self):
        return self.previous

    @property
    def previous_iteration_index(self):
        return self.previous_index

    @property
    def first_iteration(self):
        return self.first

    @property
    def is_first_iteration(self):
        return len(self) == 1

    @property
    def last_iteration(self):
        return self.last

    @property
    def last_iteration_index(self):
        return self.last_index

    @property
    def current_time_step(self):
        return self.current_iteration.current_time_step if self.current_iteration else None

    @property
    def current_time_step_index(self):
        return self.current_iteration.current_time_step_index if self.current_iteration else None

    @property
    def previous_time_step(self):
        return self.current_iteration.previous_time_step if self.current_iteration else None

    @property
    def next_time_step(self):
        return self.current_iteration.next_time_step if self.current_iteration else None

    @property
    def current_step(self):
        return self.current_iteration.current_time_step.current_step \
            if (self.current_iteration and self.current_iteration.current_time_step) else None

    @property
    def current_step_index(self):
        return self.current_iteration.current_step_index if self.current_iteration else None

    @property
    def previous_step(self):
        return self.current_iteration.current_time_step.previous_step \
            if (self.current_iteration and self.current_iteration.current_time_step
                and self.current_iteration.current_time_step.previous_step) \
            else self.initial

    @property
    def previous_step_index(self):
        return self.current_iteration.current_time_step.previous_step_index \
            if (self.current_iteration and self.current_iteration.current_time_step
                and self.current_iteration.current_time_step.previous_step) \
            else None

    @property
    def next_step(self):
        return self.current_iteration.next_step if self.current_iteration else None

    def _add_iteration(self):
        assert_condition(self.num_time_steps > 0 and self.num_nodes > 0,
                         ValueError, "Number of time steps and nodes per time step must be larger 0: NOT {}, {}"
                                     .format(self.num_time_steps, self.num_nodes),
                         self)
        self._states.append(self._element_type(num_states=self.num_nodes,
                                               num_time_steps=self.num_time_steps))


__all__ = ['IStepState', 'IStateIterator', 'IStaticStateIterator', 'ITimeStepState', 'IIterationState', 'ISolverState']
