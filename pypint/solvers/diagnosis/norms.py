# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np

from pypint.solvers.diagnosis.i_diagnosis_value import IDiagnosisValue


def supremum_norm(vec):
    """Computes uniform (or infinity) norm of given vector or :py:class:`.IDiagnosisValue`.

    Uses numpy's norm function internally.

    Parameters
    ----------
    vec : :py:class:`numpy.ndarray` or :py:class:`.IDiagnosisValue`

    Returns
    -------
    sup-norm : :py:class:`numpy.ndarray`
    """
    if isinstance(vec, float):
        return vec
    elif isinstance(vec, np.ndarray):
        return np.linalg.norm(vec, np.inf)
    elif isinstance(vec, IDiagnosisValue):
        return np.linalg.norm(vec.value, np.inf)
    else:
        return np.nan


def two_norm(vec):
    """Computes two-norm of given vector or :py:class:`.IDiagnosisValue`.

    Uses numpy's norm function internally.

    Parameters
    ----------
    vec : :py:class:`numpy.ndarray` or :py:class:`.IDiagnosisValue`

    Returns
    -------
    two-norm : :py:class:`numpy.ndarray`
    """
    if isinstance(vec, float):
        return vec
    elif isinstance(vec, np.ndarray):
        return np.linalg.norm(vec)
    elif isinstance(vec, IDiagnosisValue):
        return np.linalg.norm(vec.value)
    else:
        return np.nan


__all__ = ['supremum_norm', 'two_norm']
