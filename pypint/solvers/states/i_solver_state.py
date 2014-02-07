# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.kaltt@fz-juelich.de>
"""
from copy import deepcopy

import warnings

from pypint.solutions.data_storage import StepSolutionData, TrajectorySolutionData
from pypint.solutions import IterativeSolution
from pypint.utilities import assert_is_key, assert_condition


class IStepState(object):
    def __init__(self):
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


class IStateIterator(object):
    def __init__(self, **kwargs):
        assert_is_key(kwargs, 'solution_class', "Solution type must be given.")
        assert_is_key(kwargs, 'element_type', "Element type must be given.")
        self._solution = kwargs['solution_class']()
        self._element_type = kwargs['element_type']
        self._states = []
        self._current_index = 0

    def new(self, **kwargs):
        self._states.append(self._element_type(**kwargs))

    def done(self):
        pass

    def element_done(self, index=None):
        if index is None:
            index = self.current
        self._current_index += 1

    @property
    def current(self):
        return self[self.current_index]

    @property
    def current_index(self):
        return self._current_index

    @property
    def previous(self):
        return self[self.previous_index]

    @property
    def previous_index(self):
        if self.current_index > 0:
            return self.current_index - 1
        else:
            warnings.warn("No previous state available.")
            return None

    @property
    def next_state(self):
        if len(self) > self.current_index + 1:
            return self[self.current_index + 1]
        else:
            warnings.warn("No next state available.")
            return None

    @property
    def solution(self):
        return self._solution

    def __len__(self):
        return len(self._states)

    def __iter__(self):
        return iter(self._states)

    def __getitem__(self, item):
        return self._states[item]

    def __str__(self):
        return "{:s}({:s})".format(self.__class__.__name__, self._element_type.__name__)


class ITimeStepState(IStateIterator):
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
        super(ITimeStepState, self).__init__(**kwargs)
        del kwargs['solution_class']
        del kwargs['element_type']

        if 'num_nodes' in kwargs:
            _num_nodes = kwargs['num_nodes']
            # del kwargs['num_nodes']
            self._states = [self._element_type()] * _num_nodes

        self._delta_time_step = 0.0

    def new(self, **kwargs):
        warnings.warn("ITimeStepState.new() should not be used. NoOp.")

    def done(self):
        super(ITimeStepState, self).done()
        for _step in self:
            self.solution.add_solution(deepcopy(_step.solution))
        self.solution.finalize()

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
    def next_time_point(self):
        return self.next_state.time_point

    @property
    def current_step(self):
        return self.current

    @property
    def current_step_index(self):
        return self.current_index

    @property
    def previous_step(self):
        return self.previous

    @property
    def previous_step_index(self):
        return self.previous_index

    @property
    def time_points(self):
        return self._solution.time_points


class IIterationState(IStateIterator):
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

        if 'num_time_steps' in kwargs:
            _num_time_steps = kwargs['num_time_steps']
            del kwargs['num_time_steps']
            self._states = [self._element_type(**kwargs)] * _num_time_steps

    def new(self, **kwargs):
        warnings.warn("ITimeStepState.new() should not be used. NoOp.")

    def done(self):
        super(IIterationState, self).done()
        warnings.simplefilter("ignore", UserWarning)
        for _time_step in self:
            for _step in _time_step:
                try:
                    self.solution.add_solution(deepcopy(_step.solution))
                except ValueError:
                    # this step is already there
                    pass
        warnings.resetwarnings()
        self.solution.finalize()

    @property
    def current_time_step(self):
        return self.current

    @property
    def current_time_step_index(self):
        return self.current_index

    @property
    def current_step_index(self):
        return self.current_time_step.current_step_index

    @property
    def previous_time_step(self):
        return self.previous

    @property
    def previous_time_step_index(self):
        return self.previous_index

    @property
    def previous_step_index(self):
        return self.current_time_step.previous_step_index

    @property
    def final_step(self):
        # last time_step and last step thereof
        return self[-1][-1]

    @property
    def time_points(self):
        return self._solution.time_points


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
            kwargs['element_type'] = ITimeStepState
        super(ISolverState, self).__init__(**kwargs)

        self._num_nodes = kwargs['num_nodes'] if 'num_nodes' in kwargs else 0
        self._num_time_steps = kwargs['num_time_steps'] if 'num_time_steps' in kwargs else 0
        self._delta_interval = 0.0

    def new(self, **kwargs):
        self._states.append(self._element_type(num_nodes=self._num_nodes,
                                               num_time_steps=self._num_time_steps, **kwargs))

    def done(self):
        super(ISolverState, self).done()
        for _iteration in self:
            self.solution.add_solution(deepcopy(_iteration.solution))
        self.solution.finalize()

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
    def current_iteration(self):
        return self.current

    @property
    def current_iteration_index(self):
        return self.current_index

    @property
    def current_time_step(self):
        return self.current_iteration.current_time_step

    @property
    def current_time_step_index(self):
        return self.current_iteration.current_time_step_index

    @property
    def current_step(self):
        return self.current_iteration.current_time_step.current_step

    @property
    def current_step_index(self):
        return self.current_iteration.current_step_index

    @property
    def previous_iteration(self):
        return self.previous

    @property
    def previous_iteration_index(self):
        return self.previous_index

    @property
    def previous_time_step_index(self):
        return self.current_iteration.previous_time_step_index

    @property
    def previous_step_index(self):
        return self.current_iteration.previous_step_index

    @property
    def previous_step_index(self):
        return self.current_iteration.previous_step_index


__all__ = ['IStepState', 'IStateIterator', 'ITimeStepState', 'IIterationState', 'ISolverState']
