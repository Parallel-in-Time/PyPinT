# coding=utf-8

import numpy as np
from pypint.solutions.i_solution import ISolution
import networkx as nx

class IMGSolution(ISolution):
    """Class that saves the progress of the multigrid in a directed Graph

    """

    def __init__(self, state_class):
        # generate graph
        self.state_class = state_class
        self.graph = []

    def add_solution(self, obj, **kwargs):
        # create graph entry with fitting container
        self.graph.append(self.state_class(object, kwargs))

