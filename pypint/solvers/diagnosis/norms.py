# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import warnings

import numpy as np

from pypint.solvers.diagnosis.i_diagnosis_value import IDiagnosisValue


def supremum_norm(vec):
    """
    Summary
    -------
    Computes uniform (or infinity) norm of given vector or :py:class:`.IDiagnosisValue`.

    Extended Summary
    ----------------
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
        # warnings.warn("Unknown numeric type ('{}'). Cannot compute norm.".format(vec.__class__.__name__))
        return np.nan


def two_norm(vec):
    """
    Summary
    -------
    Computes two-norm of given vector or :py:class:`.IDiagnosisValue`.

    Extended Summary
    ----------------
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
        # warnings.warn("Unknown numeric type ('{}'). Cannot compute norm.".format(vec.__class__.__name__))
        return np.nan


__all__ = ['supremum_norm', 'two_norm']
