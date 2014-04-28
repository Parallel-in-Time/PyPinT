# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from warnings import warn
from copy import deepcopy

import numpy as np

from pypint.solvers.states.i_solver_state import IStepState, ISolverState, IStaticStateIterator
from pypint.solutions.iterative_solution import IterativeSolution
from pypint.solutions.data_storage.trajectory_solution_data import TrajectorySolutionData
from pypint.utilities import assert_condition
from pypint.utilities.logging import LOG


class MlSdcStepState(IStepState):
    def __init__(self, **kwargs):
        super(MlSdcStepState, self).__init__(**kwargs)
        self._integral = None
        self._fas_correction = None
        self._coarse_correction = None

    def has_fas_correction(self):
        return self._fas_correction is not None

    @property
    def integral(self):
        """Accessor for an integral value

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
    def coarse_correction(self):
        return self._coarse_correction

    @coarse_correction.setter
    def coarse_correction(self, coarse_correction):
        self._coarse_correction = coarse_correction

    @property
    def fas_correction(self):
        return self._fas_correction

    @fas_correction.setter
    def fas_correction(self, fas_correction):
        if not isinstance(fas_correction, np.ndarray):
            # LOG.debug("FAS Correction not given as Array. Converting it to one.")
            fas_correction = np.array([fas_correction])
        self._fas_correction = fas_correction


class MlSdcLevelState(IStaticStateIterator):
    def __init__(self, **kwargs):
        kwargs['solution_class'] = TrajectorySolutionData
        kwargs['element_type'] = MlSdcStepState
        super(MlSdcLevelState, self).__init__(**kwargs)

        self._initial = MlSdcStepState()
        self._integral = 0.0

    @property
    def initial(self):
        """Accessor for the initial value of this time step
        """
        return self._initial

    @initial.setter
    def initial(self, initial):
        self._initial = initial

    @property
    def integral(self):
        return self._integral

    @integral.setter
    def integral(self, integral):
        self._integral = integral

    @property
    def time_points(self):
        """Read-only accessor for the list of time points of this time step
        """
        return np.array([step.time_point for step in self], dtype=float)

    @property
    def values(self):
        return np.append([self.initial.value.copy()], [step.value.copy() for step in self], axis=0)

    @property
    def rhs(self):
        if self.initial.rhs_evaluated and np.all([step.rhs_evaluated for step in self]):
            return np.append([self.initial.rhs], [step.rhs for step in self], axis=0)
        else:
            return None

    @values.setter
    def values(self, values):
        assert_condition(values.shape[0] == (len(self) + 1), ValueError,
                         "Number of values does not match number of nodes: %d != %d"
                         % (values.shape[0], (len(self) + 1)),
                         checking_obj=self)
        for _step in range(0, len(self)):
            self[_step].value = values[_step + 1].copy()

    @property
    def fas_correction(self):
        _fas = np.empty(len(self) + 1, dtype=np.object)
        _fas_shape = ()
        for step_i in range(0, len(self)):
            if self[step_i].has_fas_correction():
                _fas_shape = self[step_i].fas_correction.shape
                _fas[step_i + 1] = self[step_i].fas_correction.copy()

        if len(_fas_shape) > 0:
            _fas[0] = np.zeros(_fas_shape)
            return _fas
        else:
            return None

    @fas_correction.setter
    def fas_correction(self, fas_correction):
        assert_condition(fas_correction.shape[0] == (len(self) + 1), ValueError,
                         "Number of FAS Corrections does not match number of nodes: %d != %d"
                         % (fas_correction.shape[0], (len(self) + 1)),
                         checking_obj=self)
        for _step in range(0, len(self)):
            self[_step].fas_correction = fas_correction[_step + 1].copy()

    @property
    def coarse_corrections(self):
        return np.append([np.zeros(self[0].coarse_correction.shape)], [step.coarse_correction for step in self], axis=0)

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
    def final_step(self):
        """Proxy for :py:attr:`.last`
        """
        return self.last

    @property
    def final_step_index(self):
        """Proxy for :py:attr:`.last_index`
        """
        return self.last_index


class MlSdcIterationState(IStaticStateIterator):
    def __init__(self, **kwargs):
        kwargs['solution_class'] = TrajectorySolutionData
        kwargs['element_type'] = MlSdcLevelState
        if 'num_states' in kwargs:
            warn("Levels must be initialized separately.")
            del kwargs['num_states']
        super(MlSdcIterationState, self).__init__(**kwargs)
        self._initial = None

    def add_finer_level(self, num_nodes):
        self._states.append(self._element_type(num_states=num_nodes))
        self._current_index = len(self) - 1

    def add_coarser_level(self, num_nodes):
        self._states.insert(0, self._element_type(num_states=num_nodes))
        self._current_index = len(self) - 1

    def proceed(self):
        raise RuntimeError("'proceed' is not defined in the context of different levels.")

    def finalize(self):
        pass

    def step_up(self):
        """Get to next finer level
        """
        if self.next_index:
            LOG.debug("Stepping up to level %d" % (self._current_index + 1))
            self._current_index += 1
        else:
            raise StopIteration("There is no finer level available.")

    def step_down(self):
        if not self.on_base_level:
            LOG.debug("Stepping down to level %d" % (self._current_index - 1))
            self._current_index -= 1
        else:
            raise StopIteration("There is no finer level available.")

    @property
    def time_points(self):
        """Read-only accessor for the list of time points of this time step
        """
        return np.array([step.time_point for step in self], dtype=float)

    @property
    def initial(self):
        """Accessor to the initial value of this iteration
        """
        return self._initial

    @initial.setter
    def initial(self, initial):
        self._initial = initial

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

    @property
    def on_finest_level(self):
        return self.current_level_index == len(self) - 1

    @property
    def final_step(self):
        return self.finest_level.final_step \
            if self.finest_level else None

class MlSdcSolverState(ISolverState):
    def __init__(self, **kwargs):
        kwargs['solution_class'] = IterativeSolution
        kwargs['element_type'] = MlSdcIterationState
        super(MlSdcSolverState, self).__init__(**kwargs)
        del self._num_time_steps
        self._num_level = kwargs['num_level'] if 'num_level' in kwargs else 0
        self._initial = MlSdcStepState()

    def proceed(self):
        """Proceeds to the next iteration

        Extends the sequence of :py:class:`.IIterationState` by appending a new instance with the set
        :py:attr:`.num_time_steps` and :py:attr:`.num_nodes`.
        """
        self._add_iteration()
        self._current_index = len(self) - 1
        self.current_iteration.initial = deepcopy(self.initial)

    @property
    def num_level(self):
        """Read-only accessor for the number of levels

        Returns
        -------
        num_level : :py:class:`int`
        """
        return self._num_level

    @property
    def current_level(self):
        return self.current_iteration.current_level \
            if self.current_iteration else None

    @property
    def current_level_index(self):
        return self.current_iteration.current_level_index \
            if self.current_iteration else None

    @property
    def current_step(self):
        return self.current_iteration.current_level.current_step \
            if self.current_iteration and self.current_iteration.current_level else None

    @property
    def current_step_index(self):
        return self.current_iteration.current_level.current_step_index \
            if self.current_iteration and self.current_iteration.current_level else None

    @property
    def next_step(self):
        return self.current_iteration.current_level.next_step \
            if self.current_iteration and self.current_iteration.current_level else None

    @property
    def previous_step(self):
        return self.current_iteration.current_level.previous_step \
            if self.current_iteration and self.current_iteration.current_level else None

    @property
    def previous_step_index(self):
        return self.current_iteration.current_level.previous_step_index \
            if self.current_iteration and self.current_iteration.current_level else None

    @property
    def num_time_steps(self):
        return 1 if self.current_iteration else None

    @property
    def current_time_step(self):
        return self.current_iteration.current_level \
            if self.current_iteration else None

    @property
    def current_time_step_index(self):
        return self.current_iteration.current_level_index if self.current_iteration else None

    @property
    def next_time_step(self):
        return 0 if self.current_iteration else None

    @property
    def previous_time_step(self):
        return 0 if self.current_iteration else None

    def _add_iteration(self):
        assert_condition(self.num_level > 0,
                         ValueError,
                         message="Number of number of levels and nodes per level must be larger 0: NOT {}"
                                 .format(self.num_level),
                         checking_obj=self)
        self._states.append(self._element_type(num_level=self.num_level))


__all__ = ['MlSdcStepState', 'MlSdcLevelState', 'MlSdcTimeStepState', 'MlSdcIterationState', 'MlSdcSolverState']
