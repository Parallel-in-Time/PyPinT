# coding=utf-8

import unittest
from .analyzers_tests import AnalyzersTests
from .loggers_tests import LoggersTests
from .plotters_tests import PlottersTests
from .timers_tests import TimersTests


class PluginsTests(unittest.TestSuite):
    def __init__(self):
        self.addTests(AnalyzersTests)
        self.addTests(LoggersTests)
        self.addTests(PlottersTests)
        self.addTests(TimersTests)


if __name__ == "__main__":
    unittest.main()
