# coding=utf-8

import unittest
from pypint.plugins.timers.timer_base import TimerBase


class TimerBaseTest(unittest.TestCase):
    def setUp(self):
        self.__timer = TimerBase()

    def test_start_stop_timing(self):
        self.__timer.start()
        self.__timer.stop()
        self.assertGreater(self.__timer.past(), 0.0, "Timer did not timed anything.")

    def test_start_past_timing(self):
        self.__timer.start()
        self.assertGreater(self.__timer.past(), 0.0, "Timer did not timed anything.")


if __name__ == "__main__":
    unittest.main()
