# coding=utf-8

import unittest
from .timer_base_test import TimerBaseTest


class TimersTests(unittest.TestSuite):
    def __init__(self):
        self.addTest(TimerBaseTest)


if __name__ == "__main__":
    unittest.main()
