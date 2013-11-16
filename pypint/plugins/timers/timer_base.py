# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

import time as time


class TimerBase(object):
    def __init__(self):
        self._start_time = None
        self._end_time = None

    def start(self):
        self._start_time = time.time()
        self._end_time = None

    def stop(self):
        self._end_time = time.time()

    def past(self):
        if self._end_time is None:
            self._end_time = time.time()

        return self._end_time - self._start_time
