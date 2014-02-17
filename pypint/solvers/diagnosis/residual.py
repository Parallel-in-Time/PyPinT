# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
from pypint.solvers.diagnosis.i_diagnosis_value import IDiagnosisValue


class Residual(IDiagnosisValue):
    """
    Summary
    -------
    Storage and handler of the residual of iterative time solvers.
    """


__all__ = ['Residual']
