# coding=utf-8

from pypint.plugins.plotters.single_solution_plotter import SingleSolutionPlotter
import unittest
from matplotlib import is_interactive


class SingleSolutionPlotterTest(unittest.TestCase):
    def init_with_file_name(self):
        _pltr = SingleSolutionPlotter(file_name="test_file.svg")
        self.assertFalse(is_interactive())

    def init_without_file_name(self):
        _pltr = SingleSolutionPlotter()
        self.assertTrue(is_interactive())
