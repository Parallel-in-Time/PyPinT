# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .integrator_base import IntegratorBase
from pypint.utilities import *


class SdcStep(IntegratorBase):
    def __init__(self):
        super(self.__class__, self).__init__()

    def evaluate(self, data, time_start, time_end):
        raise NotImplementedError(func_name() +
                                  "SDC integrator not yet implemented.")
