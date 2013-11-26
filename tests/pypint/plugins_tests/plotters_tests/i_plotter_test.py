# coding=utf-8

import unittest
from pypint.plugins.plotters.i_plotter import IPlotter
from matplotlib import is_interactive


class IPlotterTest(unittest.TestCase):
    def test_init_with_file_name(self):
        _pltr = IPlotter(file_name="test_file.svg")
        self.assertFalse(is_interactive())

    def test_init_without_file_name(self):
        _pltr = IPlotter()
        self.assertTrue(is_interactive())


if __name__ == "__main__":
    unittest.main()
