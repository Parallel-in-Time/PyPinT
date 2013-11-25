# coding=utf-8

import unittest
from pypint.plugins.plotters.single_solution_plotter import SingleSolutionPlotter
from matplotlib import is_interactive


class SingleSolutionPlotterTest(unittest.TestCase):
    def test_init_with_file_name(self):
        _pltr = SingleSolutionPlotter(file_name="test_file.svg")
        self.assertFalse(is_interactive())

    def test_init_without_file_name(self):
        _pltr = SingleSolutionPlotter()
        self.assertTrue(is_interactive())


if __name__ == "__main__":
    unittest.main()
