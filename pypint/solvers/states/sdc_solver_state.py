# coding=utf-8
from pypint.solvers.states.i_solver_state import IStepState, ITimeStepState, IIterationState, ISolverState
from pypint.solutions.iterative_solution import IterativeSolution
from pypint.solutions.data_storage import TrajectorySolutionData


class SdcStepState(IStepState):
    def __init__(self, **kwargs):
        super(SdcStepState, self).__init__(**kwargs)
        self._integral = 0.0

    @property
    def integral(self):
        return self._integral
    @integral.setter
    def integral(self, integral):
        self._integral = integral


class SdcTimeStepState(ITimeStepState):
    def __init__(self, **kwargs):
        kwargs['solution_class'] = TrajectorySolutionData
        kwargs['element_type'] = SdcStepState
        super(SdcTimeStepState, self).__init__(**kwargs)


class SdcIterationState(IIterationState):
    def __init__(self, **kwargs):
        kwargs['solution_class'] = TrajectorySolutionData
        kwargs['element_type'] = SdcTimeStepState
        super(SdcIterationState, self).__init__(**kwargs)


class SdcSolverState(ISolverState):
    def __init__(self, **kwargs):
        kwargs['solution_class'] = IterativeSolution
        kwargs['element_type'] = SdcIterationState
        super(SdcSolverState, self).__init__(**kwargs)
        self._initial_state = SdcStepState()


__all__ = ['SdcStepState', 'SdcTimeStepState', 'SdcIterationState', 'SdcSolverState']
