# coding=utf-8


from pypint.solutions.iterative_solution import IterativeSolution
import copy

class MGLevelFullIterativeSolution(IterativeSolution):
    """Saves the progress of the local level after each smoother step

    """
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.iterative_list = []

    def add_solution(self, level, *args, **kwargs ):
        """Adds a new solution storage object from a IMultigridLevel object

        """
        self.iterative_list.append(copy.copy(level.mid))
        self._used_iterations += 1
