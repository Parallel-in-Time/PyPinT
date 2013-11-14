# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

import time as time


class TimerBase(object):
    def __init__(self):
        self.__start_time = None
        self.__end_time = None

    def start(self):
        self.__start_time = time.time()
        self.__end_time = None

    def stop(self):
        self.__end_time = time.time()

    def past(self):
        if self.__end_time is not None:
            self.__end_time = time.time()

        return self.__end_time - self.__start_time
