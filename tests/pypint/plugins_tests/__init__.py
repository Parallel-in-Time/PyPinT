# coding=utf-8

import unittest
from .analyzers_tests import *
from .loggers_tests import *
from .plotters_tests import *
from .timers_tests import *

class PluginsTests(unittest.TestSuite):
    def __init__(self):
        self.addTests(AnalyzersTests)
        self.addTests(LoggersTests)
        self.addTests(PlottersTests)
        self.addTests(TimersTests)


if __name__ == "__main__":
    unittest.main()

__all__ = ["PluginsTests"]
