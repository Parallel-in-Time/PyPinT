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
    """State of a single integration step

    A integration step is a single point in time.
    """
    def __init__(self, **kwargs):
        self._solution = StepSolutionData()
        self._delta_tau = 0.0

    def done(self):
        """Finalize this state and its included solution
        """
        self.solution.finalize()

    @property
    def solution(self):
        """Proxy to the included solution of the state

        Returns
        -------
        solution : :py:class:`.StepSolutionData`
        """
        return self._solution

    @property
    def time_point(self):
        """Proxy for :py:attr:`.StepSolutionData.time_point`
        """
        return self._solution.time_point

    @property
    def delta_tau(self):
        """Accessor for the width of the integration step

        Usually the distance to the previous integration node.

        Parameters
        ----------
        delta_tau : :py:class:`float`

        Returns
        -------
        delta_tau : :py:class:`float`

        Raises
        ------
        ValueError
            if ``delta_tau`` is not non-zero positive
        """
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
    """Interface for a sequence of states

    :py:class:`.IStateIterator` models a sequence of states allowing for easy iteration over all containing states.

    Examples
    --------
    >>> from pypint.solutions.data_storage import TrajectorySolutionData
    >>> my_states = IStateIterator(solution_class=TrajectorySolutionData, element_type=IStepState, num_states=3)
    >>> len(my_states)
    3

    Notes
    -----
    Please keep in mind, that only derived classes can alter the currently accesable state.
    Such a class is :py:class:`.IStaticStateIterator`.
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        solution_class : :py:class:`.ISolution`, :py:class:`.StepSolutionData` or :py:class:`.TrajectorySolutionData`

        element_type : :py:class:`.IStepState` or :py:class:`.IStateItertor`

        num_states : :py:class:`int`
            *(optional)*

        Raises
        ------
        ValueError
            if ``num_states`` is not a non-zero positive integer
        """
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
        """Finalize this sequence of states

        This copies the solution data from all containing states to its own solution object and finalizes it.
        As well, the :py:attr:`.current_index` is reset to zero.

        Raises
        ------
        RuntimeError
            if this state has already been finalized
        """
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
        """Read-only accessor to determine the finalized state of this state sequence

        Raises
        ------
        RuntimeError
            if this object is finalized but its containing solution object is not
            (though, this should never happen)
        """
        if self._finalized:
            # if this throws, something is really broken
            assert_condition(self.solution.finalized,
                             RuntimeError, "State is finalized but not its solution object.",
                             self)
        return self._finalized

    @property
    def solution(self):
        """Accessor for the containing solution object
        """
        return self._solution

    @property
    def current(self):
        """Accessor to the current state of this sequence
        """
        return self[self.current_index] if self.current_index is not None else None

    @property
    def current_index(self):
        """Index of the current state of this sequence
        """
        return self._current_index if len(self) > self._current_index else None

    @property
    def previous(self):
        """Accessor to the previous state of this sequence

        Returns
        -------
        previous : :py:class:`.IStepState`, :py:class:`.IStateIterator` or :py:class:`None`
            :py:class:`None` if :py:attr:`.previous_index` returns :py:class:`None`
        """
        return self[self.previous_index] if self.previous_index is not None else None

    @property
    def previous_index(self):
        """Index of the previous state of this sequence

        Returns
        -------
        previous_index : :py:class:`int` or :py:class:`None`
            :py:class:`None` if no previous state is available, i.e. if :py:attr:`.current_index` is 0
        """
        return self.current_index - 1 if self.current_index is not None and self.current_index > 0 else None

    @property
    def first(self):
        """Accessor to the first state of this sequence

        Returns
        -------
        first : :py:class:`None`
            if :py:attr:`.first_index` returns :py:class:`None`
        """
        return self[self.first_index] if self.first_index is not None else None

    @property
    def first_index(self):
        """Index of the first state of this seuqence

        Returns
        -------
        first_index : ``0`` or :py:class:`None`
            :py:class:`None` is returned, if there are no states in this sequence
        """
        return 0 if len(self) > 0 else None

    @property
    def last(self):
        """Accessor for the last state in this sequence

        Returns
        -------
        last : :py:class:`.IStepState`, :py:class:`.IStateIterator` or :py:class:`None`
            :py:class:`None` if :py:attr:`.last_index` is :py:class:`None`
        """
        return self[self.last_index] if self.last_index is not None else None

    @property
    def last_index(self):
        """Index of the last state in this sequence

        Returns
        -------
        last_index : :py:class:`int` or :py:class:`None`
            :py:class:`None` if there are no states in this sequence
        """
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
    """Specialized sequence of states with fixed number of states
    """
    def proceed(self):
        """Proceed :py:attr:`.current` to the next state in the sequence

        Raises
        ------
        RuntimeError
            if this sequence has already been finalized via :py:meth:`.finalize`
        """
        assert_condition(not self.finalized,
                         RuntimeError, "This {} is already done.".format(self.__class__.__name__),
                         self)
        if self.next_index is not None:
            self._current_index += 1
        else:
            raise StopIteration("No further states available.")

    @property
    def next(self):
        """Accessor for the next state in this sequence

        Returns
        -------
        next : :py:class:`.IStepState`, :py:class:`.IStateIterator` or :py:class:`None`
            :py:class:`None` if :py:attr:`.next_index` is :py:class:`None`
        """
        return self[self.next_index] if self.next_index is not None else None

    @property
    def next_index(self):
        """Index of the next state in this sequence

        Returns
        -------
        next_index : :py:class:`int` or :py:class:`None`
            :py:class:`None` if :py:attr:`.current` is already the last state in this sequence
        """
        return self.current_index + 1 \
            if self.current_index is not None and len(self) > self.current_index + 1 else None


class ITimeStepState(IStaticStateIterator):
    """Stores integration step states of a single time step.
    """
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        num_states : :py:class:`int`
            number of states in this sequence
        solution_class : :py:class:`.TrajectorySolutionData` or :py:class:`.StepSolutionData`
            *(optional)*
            defaults to :py:class:`.TrajectorySolutionData`
        element_type : :py:class:`.IStepState` or :py:class:`.IStateIterator`
            *(optional)*
            defaults to :py:class:`.IStepState`

        Raises
        ------
        ValueError
            if ``num_states`` is not given
        """
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
        """Accessor for the width of the time step

        Parameters
        ----------
        delta_time_step : :py:class:`float`

        Returns
        -------
        width_of_time_step : :py:class:`float`

        Raises
        ------
        ValueError
            *(only setter)*
            if ``delta_time_step`` is not a non-zero positive :py:class:`float`
        """
        return self._delta_time_step

    @delta_time_step.setter
    def delta_time_step(self, delta_time_step):
        assert_condition(delta_time_step > 0.0,
                         ValueError, "Delta interval must be non-zero positive: NOT {}".format(delta_time_step),
                         self)
        self._delta_time_step = delta_time_step

    @property
    def initial(self):
        """Accessor for the initial value of this time step
        """
        return self._initial

    @initial.setter
    def initial(self, initial):
        self._initial = initial

    @property
    def time_points(self):
        """Read-only accessor for the list of time points of this time step
        """
        return np.array([step.time_point for step in self], dtype=float)

    @property
    def current_time_point(self):
        """Accessor for the current step's time point

        Returns
        -------
        current_time_point : :py:class:`float` or :py:class:`None`
            :py:class:`None` is returned if :py:attr:`.current_step` is :py:class:`None`
        """
        return self.current_step.time_point if self.current_step is not None else None

    @property
    def previous_time_point(self):
        """Accessor for the previous step's time point

        Returns
        -------
        previous_time_point : :py:class:`float` or :py:class:`None`
            :py:class:`None` is returned if :py:attr:`.previous_step` is :py:class:`None`
        """
        return self.previous_step.time_point if self.previous is not None else None

    @property
    def next_time_point(self):
        """Accessor for the next step's time point

        Returns
        -------
        next_time_point : :py:class:`float` or :py:class:`None`
            :py:class:`None` is returned if :py:attr:`.next_step` is :py:class:`None`
        """
        return self.next.time_point if self.next_step is not None else None

    @property
    def current_step(self):
        """Proxy for :py:attr:`.current`
        """
        return self.current

    @property
    def current_step_index(self):
        """Proxy for :py:attr:`.current_index`
        """
        return self.current_index

    @property
    def previous_step(self):
        """Accessor for the previous step

        Returns
        -------
        previous step : :py:class:`.IStepState` or :py:class:`None`
            :py:class:`None` is returned if :py:attr:`.previous_index` is :py:class:`None`
        """
        return self.previous if self.previous_index is not None else self.initial

    @property
    def previous_step_index(self):
        """Proxy for :py:attr:`.previous_index`
        """
        return self.previous_index

    @property
    def next_step(self):
        """Proxy for :py:attr:`.next`
        """
        return self.next

    @property
    def next_step_index(self):
        """Proxy for :py:attr:`.next_index`
        """
        return self.next_index

    @property
    def last_step(self):
        """Proxy for :py:attr:`.last`
        """
        return self.last

    @property
    def last_step_index(self):
        """Proxy for :py:attr:`.last_index`
        """
        return self.last_index


class IIterationState(IStaticStateIterator):
    """Stores time step states of a single iteration.
    """
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        num_time_steps : :py:class:`int`
            number of time steps in this sequence
        num_states : :py:class:`int`
            number of steps per time step
        solution_class : :py:class:`.TrajectorySolutionData`, *any other solution class*
            *(optional)*
            defaults to :py:class:`.TrajectorySolutionData`
        element_type : :py:class:`.IStateIterator`
            *(optional)*
            defaults to :py:class:`.ITimeStepState`

        Raises
        ------
        ValueError
            if ``num_time_steps`` is not given
        """
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
        self._initial = None

    def finalize(self):
        """Finalizes this iteration and copies solutions

        The solutions of all steps of all time steps are copied to this sequence's :py:class:`.TrajectorySolutionData`
        and is finalized afterwards.

        The remaining behaviour is the same as the overridden method.

        See Also
        --------
        :py:meth:`.IStateIterator.finalize` : overridden method
        """
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
        """Proceeds to the next time step

        Same as :py:meth:`.IStaticStateIterator.proceed` with the addition, that the current time step's initial
        value is set as a reference to the previous time step's last step.
        """
        super(IIterationState, self).proceed()  # -> current_index += 1
        # link initial step of this time step to the previous' last step
        self.current_time_step.initial = self.previous_time_step.last_step

    @property
    def initial(self):
        """Accessor to the initial value of this iteration
        """
        return self._initial

    @initial.setter
    def initial(self, initial):
        self._initial = initial

    @property
    def current_time_step(self):
        """Proxy for :py:attr:`.current`
        """
        return self.current

    @property
    def current_time_step_index(self):
        """Proxy for :py:attr:`.current_index`
        """
        return self.current_index

    @property
    def previous_time_step(self):
        """Proxy for :py:attr:`.previous`
        """
        return self.previous

    @property
    def previous_time_step_index(self):
        """Proxy for :py:attr:`.previous_index`
        """
        return self.previous_index

    @property
    def next_time_step(self):
        """Proxy for :py:attr:`.next`
        """
        return self.next

    @property
    def next_time_step_index(self):
        """Proxy for :py:attr:`.next_index`
        """
        return self.next_index

    @property
    def first_time_step(self):
        """Proxy for :py:attr:`.first`
        """
        return self.first

    @property
    def first_time_step_index(self):
        """Proxy for :py:attr:`.first_index`
        """
        return self.first_index

    @property
    def last_time_step(self):
        """Proxy for :py:attr:`.last`
        """
        return self.last

    @property
    def last_time_step_index(self):
        """Proxy for :py:attr:`.last_index`
        """
        return self.last_index

    @property
    def current_step(self):
        """Proxy for :py:attr:`.ITimeStepState.current_step`
        """
        return self.current_time_step.current_step

    @property
    def current_step_index(self):
        """Proxy for :py:attr:`.ITimeStepState.current_step_index`
        """
        return self.current_time_step.current_step_index

    @property
    def previous_step(self):
        """Read-only accessor for the previous step

        Returns
        -------
        previous_step : :py:class:`.IStepState`
            proxies :py:attr:`.ITimeStepState.previous_step` if it is not :py:class:`None`
            else :py:attr:`.ITimeStepState.initial`
        """
        return self.current_time_step.previous_step \
            if self.current_time_step.previous_step is not None else self.first_time_step.initial

    @property
    def next_step(self):
        """Read-only accessor for the next step

        Returns
        -------
        next_step : :py:class:`.IStepState`
            proxies :py:attr:`.ITimeStepState.next_step` if it is not :py:class:`None`
            else returns :py:attr:`.ITimeStepState.first` if :py:attr:`.next_time_step` is not :py:class:`None`
            else returns :py:class:`None`
        """
        return self.current_time_step.next_step \
            if self.current_time_step.next_step is not None \
            else (self.next_time_step.first if self.next_time_step is not None else None)

    @property
    def first_step(self):
        """Read-only accessor for the first step

        Returns
        -------
        first_step : :py:class:`.IStepState`
            proxies :py:attr:`.ITimeStepState.first` if :py:attr:`.first_time_step` is not :py:class:`None`
            else returns :py:class:`None`
        """
        return self.first_time_step.first if self.first_time_step is not None else None

    @property
    def final_step(self):
        """Read-only accessor for the very last step

        Returns
        -------
        last_step : :py:class:`.IStepState`
            proxies :py:attr:`.ITimeStepState.last` if :py:attr:`.last_time_step` is not :py:class:`None`
            else returns :py:class:`None`
        """
        return self.last_time_step.last if self.last_time_step is not None else None

    @property
    def time_points(self):
        """Read-only accessor for all time points

        Returns
        -------
        time_points : :py:class:`numpy.array(dtype=float)`
        """
        return np.array([_step.time_point for _time_point in self for _step in _time_point], dtype=float)


class ISolverState(IStateIterator):
    """Stores iteration states.
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
        """Proceeds to the next iteration

        Extends the sequence of :py:class:`.IIterationState` by appending a new instance with the set
        :py:attr:`.num_time_steps` and :py:attr:`.num_nodes`.
        """
        self._add_iteration()
        self._current_index = len(self) - 1
        self.current_iteration.initial = self.initial
        self.current_iteration.first_time_step.initial = self.current_iteration.initial

    def finalize(self):
        """Finalizes the whole solver state.

        This copies the :py:class:`.TrajectorySolutionData` objects from the :py:class:`.IIterationState` instances of
        this sequence to the main :py:class:`.IterativeSolution` object and finalizes it.
        """
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
        """Read-only accessor for the number of nodes per time step.
        """
        return self._num_nodes

    @property
    def num_time_steps(self):
        """Read-only accessor for the number of time steps per iteration.
        """
        return self._num_time_steps

    @property
    def delta_interval(self):
        """Accessor for the total interval width.

        Parameters
        ----------
        delta_interval : :py:class:`float`
            width of the whole interval

        Raises
        ------
        ValueError
            if given interval is not a non-zero float
        """
        return self._delta_interval

    @delta_interval.setter
    def delta_interval(self, delta_interval):
        assert_condition(delta_interval > 0.0,
                         ValueError, "Delta interval must be non-zero positive: NOT {}".format(delta_interval),
                         self)
        self._delta_interval = delta_interval

    @property
    def initial(self):
        """Accessor for the initial value
        """
        return self._initial

    @initial.setter
    def initial(self, initial):
        self._initial = initial

    @property
    def current_iteration(self):
        """Proxies :py:attr:`.IStateIterator.current`
        """
        return self.current

    @property
    def current_iteration_index(self):
        """Proxies :py:attr:`.IStateIterator.current_index`
        """
        return self.current_index

    @property
    def previous_iteration(self):
        """Proxies :py:attr:`.IStateIterator.previous`
        """
        return self.previous

    @property
    def previous_iteration_index(self):
        """Proxies :py:attr:`.IStateIterator.previous_index`
        """
        return self.previous_index

    @property
    def first_iteration(self):
        """Proxies :py:attr:`.IStateIterator.first`
        """
        return self.first

    @property
    def is_first_iteration(self):
        """Check on whether current iteration is the first one.

        Returns
        -------
        is_first : :py:class:`bool`

            :py:class:`True`
                if ``len(self)`` is one
            :py:class:`False`
                otherwise
        """
        return len(self) == 1

    @property
    def last_iteration(self):
        """Proxies :py:attr:`.IStateIterator.last`
        """
        return self.last

    @property
    def last_iteration_index(self):
        """Proxies :py:attr:`.IStateIterator.last_index`
        """
        return self.last_index

    @property
    def current_time_step(self):
        """Read-only accessor for the current time step

        Proxies :py:attr:`.IIterationState.current_time_step` if :py:attr:`.current_iteration` is not :py:class:`None`
        else returns :py:class:`None`.
        """
        return self.current_iteration.current_time_step \
            if self.current_iteration is not None else None

    @property
    def current_time_step_index(self):
        """Read-only accessor for the current time step's index

        Proxies :py:attr:`.IIterationState.current_time_step_index` if :py:attr:`.current_iteration` is not
        :py:class:`None` else returns :py:class:`None`.
        """
        return self.current_iteration.current_time_step_index \
            if self.current_iteration is not None else None

    @property
    def previous_time_step(self):
        """Read-only accessor for the revious time step

        Proxies :py:attr:`.IIterationState.previous_time_step` if :py:attr:`.current_iteration` is not :py:class:`None`
        else returns :py:class:`None`.
        """
        return self.current_iteration.previous_time_step \
            if self.current_iteration is not None else None

    @property
    def next_time_step(self):
        """Read-only accessor for the next time step

        Proxies :py:attr:`.IIterationState.next_time_step` if :py:attr:`.current_iteration` is not :py:class:`None`
        else returns :py:class:`None`.
        """
        return self.current_iteration.next_time_step \
            if self.current_iteration is not None else None

    @property
    def current_step(self):
        """Read-only accessor for the current step

        Proxies :py:attr:`.ITimeStepState.current_step` if :py:attr:`.current_iteration` and
        :py:attr:`.IIterationState.current_time_step` are not :py:class:`None` else returns :py:class:`None`.
        """
        return self.current_iteration.current_time_step.current_step \
            if (self.current_iteration is not None and self.current_iteration.current_time_step is not None) \
            else None

    @property
    def current_step_index(self):
        """Read-only accessor for the current step's index

        Proxies :py:attr:`.ITimeStepState.current_step_index` if :py:attr:`.current_iteration` and
        :py:attr:`.IIterationState.current_time_step` are not :py:class:`None` else returns :py:class:`None`.
        """
        return self.current_iteration.current_time_step.current_step_index \
            if (self.current_iteration is not None and self.current_iteration.current_time_step is not None) \
            else None

    @property
    def previous_step(self):
        """Read-only accessor for the previous step

        Proxies :py:attr:`.ITimeStepState.previous_step` if neither :py:attr:`.current_iteration` nor
        :py:attr:`.IIterationState.current_time_step` or :py:attr:`.IIterationState.previous_step` are :py:class:`None`
        else returns :py:class:`None`.
        """
        return self.current_iteration.current_time_step.previous_step \
            if (self.current_iteration is not None and self.current_iteration.current_time_step is not None
                and self.current_iteration.current_time_step.previous_step is not None) \
            else self.initial

    @property
    def previous_step_index(self):
        """Read-only accessor for the previous step's index

        Proxies :py:attr:`.ITimeStepState.previous_step_index` if neither :py:attr:`.current_iteration` nor
        :py:attr:`.IIterationState.current_time_step` or :py:attr:`.IIterationState.previous_step` are :py:class:`None`
        else returns :py:class:`None`.
        """
        return self.current_iteration.current_time_step.previous_step_index \
            if (self.current_iteration is not None and self.current_iteration.current_time_step is not None
                and self.current_iteration.current_time_step.previous_step is not None) \
            else None

    @property
    def next_step(self):
        """Read-only accessor for the next step

        Proxies :py:attr:`.IIterationState.next_step` if :py:attr:`.current_iteration` is not :py:class:`None` else
        returns :py:class:`None`.
        """
        return self.current_iteration.next_step if self.current_iteration is not None else None

    def _add_iteration(self):
        assert_condition(self.num_time_steps > 0 and self.num_nodes > 0,
                         ValueError, "Number of time steps and nodes per time step must be larger 0: NOT {}, {}"
                                     .format(self.num_time_steps, self.num_nodes),
                         self)
        self._states.append(self._element_type(num_states=self.num_nodes,
                                               num_time_steps=self.num_time_steps))


__all__ = ['IStepState', 'IStateIterator', 'IStaticStateIterator', 'ITimeStepState', 'IIterationState', 'ISolverState']
