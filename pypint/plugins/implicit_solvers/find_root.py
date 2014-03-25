# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np
from scipy.optimize import root

from pypint.utilities import assert_is_callable, assert_is_instance


def find_root(fun, x0, method="hybr"):
    """Wrapper around SciPy's generic root finding algorithm to support complex numbers.

    SciPy's generic root finding algorithm (``scipy.optimize.root``) is not able to deal with functions returning
    and/or accepting arrays with complex numbers.

    This wrapped call will first convert all arrays of complex numbers into arrays of floats while splitting each
    complex number up into two floats.

    Parameters
    ----------
    fun : :py:class:`callable`
        Complex function to find the root of

    x0 : :py:class:`numpy.ndarray`
        Initial guess.

    method : :py:class:`str`
        Root finding method to be used.
        See ``scipy.optimize.root`` for details.

    Returns
    -------
    Same solution object as ``scipy.optimize.root`` but with ``x`` being converted back including complex numbers.

    Examples
    --------
        from pypint.plugins.implicit_solvers.find_root import find_root
        import numpy
        fun = lambda x: (-1.0 + 1.0j) * x
        sol = find_root(fun, numpy.array([0.0]))
    """
    assert_is_instance(x0, np.ndarray, descriptor="Initial Guess")
    assert_is_callable(fun, descriptor="Function to find root of")
    assert_is_instance(method, str, descriptor="Root finding method")

    _value_map = {}
    _transformed_size = 0
    _transform_necessary = False
    for i in range(0, x0.size):
        if isinstance(x0[i], complex):
            _value_map[i] = [_transformed_size, _transformed_size + 1]
            _transformed_size += 2
            _transform_necessary = True
        else:
            _value_map[i] = [_transformed_size]
            _transformed_size += 1

    if _transform_necessary:
        _wrapped_func = \
            lambda x_next: _transform_to_real(fun(_transform_to_complex(x_next, _value_map)), _value_map, _transformed_size)
        sol = root(fun=_wrapped_func, x0=_transform_to_real(x0, _value_map, _transformed_size), method=method)
    else:
        sol = root(fun=fun, x0=x0, method=method)

    if sol.success and _transform_necessary:
        sol.x = _transform_to_complex(sol.x, _value_map)
    return sol


def _transform_to_real(x_complex, value_map, transformed_size):
    _x_real = np.zeros(transformed_size, dtype=np.float)
    for elem in value_map:
        if len(value_map[elem]) == 2:
            _x_real[value_map[elem][0]] = x_complex[elem].real
            _x_real[value_map[elem][1]] = x_complex[elem].imag
        else:
            _x_real[value_map[elem][0]] = x_complex[elem].real
    return _x_real


def _transform_to_complex(x_real, value_map):
    _x_complex = np.zeros(len(value_map), dtype=np.complex)
    for elem in value_map:
        if len(value_map[elem]) == 2:
            _x_complex[elem] = complex(*x_real[value_map[elem]])
        else:
            _x_complex[elem] = x_real[value_map[elem][0]]
    return _x_complex


__all__ = ['find_root']
