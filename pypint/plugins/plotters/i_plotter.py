# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""


class IPlotter(object):
    def __init__(self, *args, **kwargs):
        self._file_name = None

    def plot(self, *args, **kwargs):
        pass
