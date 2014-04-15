# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np

from pypint.solvers.states.i_solver_state import IStepState, ITimeStepState, IIterationState, ISolverState, IStaticStateIterator
from pypint.solutions.iterative_solution import IterativeSolution
from pypint.solutions.data_storage.trajectory_solution_data import TrajectorySolutionData
from pypint.utilities import assert_condition


class MlSdcStepState(IStepState):
    def __init__(self, **kwargs):
        super(MlSdcStepState, self).__init__(**kwargs)
        self._integral = 0.0
        self._fas_correction = 0.0

    @property
    def integral(self):
        """Accessor for an integral value

        Defaults to :math:`0.0` if not set.

        Parameters
        ----------
        integral : :py:class:`float`
            (no consistency checks are done)
        """
        return self._integral

    @integral.setter
    def integral(self, integral):
        self._integral = integral

    @property
    def fas_correction(self):
        return self._fas_correction

    @fas_correction.setter
    def fas_correction(self, fas_correction):
        self._fas_correction = fas_correction


class MlSdcLevelState(IStaticStateIterator):
    def __init__(self, **kwargs):
        kwargs['solution_class'] = TrajectorySolutionData
        kwargs['element_type'] = MlSdcStepState
        super(MlSdcLevelState, self).__init__(**kwargs)

        self._initial = MlSdcStepState()

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


class MlSdcTimeStepState(IStaticStateIterator):
    """

    The finer the level, the higher the index number.
    The coarsest level has index 0.
    """

    def __init__(self, **kwargs):
        kwargs['solution_class'] = TrajectorySolutionData
        kwargs['element_type'] = MlSdcLevelState
        super(MlSdcTimeStepState, self).__init__(**kwargs)

        self._delta_time_step = 0.0
        self._initial = MlSdcStepState()

    # there is no context as proceeding; only step-up and step-down
    proceed = None

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
        assert_condition(delta_time_step > 0.0, ValueError,
                         message="Delta interval must be non-zero positive: NOT {}".format(delta_time_step),
                         checking_obj=self)
        self._delta_time_step = delta_time_step

    @property
    def initial(self):
        """Accessor for the initial value of this time step
        """
        return self._initial

    @initial.setter
    def initial(self, initial):
        self._initial = initial

    def step_up(self):
        """Get to next finer level
        """
        if self.current_level_index < len(self) - 1:
            self._current_index += 1
        else:
            raise StopIteration("There is no finer level available.")

    def step_down(self):
        if not self.on_base_level:
            self._current_index -= 1
        else:
            raise StopIteration("There is no finer level available.")

    @property
    def current_level(self):
        return self.current

    @property
    def current_level_index(self):
        return self.current_index

    @property
    def finer_level(self):
        return self.next

    @property
    def finer_level_index(self):
        return self.next_index

    @property
    def coarser_level(self):
        return self.previous

    @property
    def coarser_level_index(self):
        return self.previous_index

    @property
    def finest_level(self):
        return self.last

    @property
    def finest_level_index(self):
        return self.last_index

    @property
    def coarsest_level(self):
        return self.first

    @property
    def coarsest_level_index(self):
        return self.first_index

    @property
    def base_level(self):
        return self.coarsest_level()

    @property
    def on_base_level(self):
        return self.current_level_index == 0


class MlSdcIterationState(IIterationState):
    def __init__(self, **kwargs):
        kwargs['solution_class'] = TrajectorySolutionData
        kwargs['element_type'] = MlSdcTimeStepState
        super(MlSdcIterationState, self).__init__(**kwargs)


class MlSdcSolverState(ISolverState):
    def __init__(self, **kwargs):
        kwargs['solution_class'] = IterativeSolution
        kwargs['element_type'] = MlSdcIterationState
        super(MlSdcSolverState, self).__init__(**kwargs)
        self._num_level = kwargs['num_level'] if 'num_level' in kwargs else 0
        self._initial_state = MlSdcStepState()

    @property
    def num_level(self):
        """Read-only accessor for the number of levels

        Returns
        -------
        num_level : :py:class:`int`
        """
        return self._num_level

    def _add_iteration(self):
        assert_condition(self.num_time_steps > 0 and self.num_nodes > 0 and self.num_level > 0,
                         ValueError,
                         message="Number of time steps, number of levels and nodes per time step must be larger 0:"
                                 "NOT {}, {}, {}"
                                 .format(self.num_time_steps, self.num_level, self.num_nodes),
                         checking_obj=self)
        self._states.append(self._element_type(num_states=self.num_nodes,
                                               num_level=self.num_level,
                                               num_time_steps=self.num_time_steps))


__all__ = ['MlSdcStepState', 'MlSdcLevelState', 'MlSdcTimeStepState', 'MlSdcIterationState', 'MlSdcSolverState']
