# coding=utf-8
"""
MultigridSolution
"""
from pypint.solutions.i_solution import ISolution


class MultiGridSolution(ISolution):
    """
    Summary
    _______
    Saves the computed data accordingly
    """
    def __init__(self):
        super(self.__class__, self).__init__()
        self._used_smoothing_operations = 0
        self._used_levels = None
        self._used_cycles = {"w": 0, "v": 0, "FMG": 0}
        pass

    def add_solution(self, *args, **kwargs):
        """ A


        """

