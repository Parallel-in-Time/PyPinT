# coding=utf-8
"""
.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

import numpy as np

from pypint.solvers.diagnosis.i_diagnosis_value import IDiagnosisValue
from pypint.utilities import assert_is_instance


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

    assert_is_instance(vec, (np.ndarray, IDiagnosisValue),
                       "The infinity norm requires a numpy.ndarray or IDiagnosisValue: NOT {}"
                       .format(vec.__class__.__name__))
    return \
        np.linalg.norm(vec, np.inf) if isinstance(vec, np.ndarray) else np.linalg.norm(vec.value, np.inf)


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

    assert_is_instance(vec, (np.ndarray, IDiagnosisValue),
                       "The infinity requires a numpy.ndarray or IDiagnosisValue: NOT {}"
                       .format(vec.__class__.__name__))
    return \
        np.linalg.norm(vec) if isinstance(vec, np.ndarray) else np.linalg.norm(vec.value)


__all__ = ['supremum_norm', 'two_norm']
