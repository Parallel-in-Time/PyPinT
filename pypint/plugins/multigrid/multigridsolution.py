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
