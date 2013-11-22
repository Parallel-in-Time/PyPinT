# coding=utf-8
"""

.. moduleauthor: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class IProblem(object):
    def __init__(self, *args, **kwargs):
        if "function" in kwargs:
            self._function = kwargs["function"]
        else:
            self._exact = None

        if "time_start" in kwargs:
            self._time_start = kwargs["time_start"]
        else:
            self._time_start = None

        if "time_end" in kwargs:
            self._time_end = kwargs["time_end"]
        else:
            self._time_end = None

        if "exact_function" in kwargs:
            self._exact_function = kwargs["exact_function"]
        else:
            self._exact_function = None

    def evaluate(self, time, phi_of_time):
        return self.function(time, phi_of_time)

    def exact(self, time, phi_of_time):
        return self.exact_function(time, phi_of_time)

    def has_exact(self):
        return self.exact_function is not None

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, function):
        self._function = function

    @property
    def time_start(self):
        return self._time_start

    @time_start.setter
    def time_start(self, time_start):
        self._time_start = time_start

    @property
    def time_end(self):
        return self._time_end

    @time_end.setter
    def time_end(self, time_end):
        self._time_end = time_end

    @property
    def exact_function(self):
        return self._exact_function

    @exact_function.setter
    def exact_function(self, exact_function):
        self._exact_function = exact_function

    def __str__(self):
        return "{:s} on [{:.3f}, {:.3f}]".format(self.__class__.__name__, self.time_start, self.time_end)
