# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.solvers.diagnosis.i_diagnosis_value import IDiagnosisValue


class Error(IDiagnosisValue):
    """Storage and handler of the approximation error of iterative time solvers.
    """


__all__ = ['Error']
