# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

from .i_iterative_time_solver import IIterativeTimeSolver
from pypint.integrators.sdc_integrator import SdcStep
from pypint.utilities import func_name


class Sdc(IIterativeTimeSolver):
    def __init__(self):
        super(self.__class__, self).__init__()

    def init(self, problem, integrator=SdcStep(), *args, **kwargs):
        super(self.__class__, self).init(problem, integrator)

    def run(self):
        raise NotImplementedError(func_name() +
                                  "SDC algorithm not yet implemented.")
