# coding=utf-8

from pypint.plugins.plotters.i_plotter import IPlotter
import unittest
from matplotlib import is_interactive


class IPlotterTest(unittest.TestCase):
    def init_with_file_name(self):
        _pltr = IPlotter(file_name="test_file.svg")
        self.assertFalse(is_interactive())

    def init_without_file_name(self):
        _pltr = IPlotter()
        self.assertTrue(is_interactive())
