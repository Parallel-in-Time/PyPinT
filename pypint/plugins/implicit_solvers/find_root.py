# coding=utf-8
"""

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import numpy as np
from scipy.optimize import root
from pypint.utilities.tracing import func_name


def find_root(fun, x0, method="hybr"):
    """
    Summary
    -------

    Parameters
    ----------
    """
    if not isinstance(x0, np.ndarray):
        raise ValueError(func_name() +
                         "Initial start value must be a numpy.ndarray.")
    if not callable(fun):
        raise ValueError(func_name() +
                         "Function to find root of must be callable.")
    if not isinstance(method, str):
        raise ValueError(func_name() +
                         "Root finding method must be defined as a string.")

    _value_map = {}
    _transformed_size = 0
    for i in range(0, x0.size):
        if isinstance(x0[i], complex):
            _value_map[i] = [_transformed_size, _transformed_size + 1]
            _transformed_size += 2
        else:
            _value_map[i] = [_transformed_size]
            _transformed_size += 1

    _wrapped_func = \
        lambda x_next: transform_to_real(fun(transform_to_complex(x_next, _value_map)), _value_map, _transformed_size)

    sol = root(fun=_wrapped_func, x0=transform_to_real(x0, _value_map, _transformed_size), method=method)
    if sol.success:
        sol.x = transform_to_complex(sol.x, _value_map)
    return sol


def transform_to_real(x_complex, value_map, transformed_size):
    _x_real = np.zeros(transformed_size, dtype=np.float)
    for elem in value_map:
        if len(value_map[elem]) == 2:
            _x_real[value_map[elem][0]] = x_complex[elem].real
            _x_real[value_map[elem][1]] = x_complex[elem].imag
        else:
            _x_real[value_map[elem][0]] = x_complex[elem].real
    return _x_real


def transform_to_complex(x_real, value_map):
    _x_complex = np.zeros(len(value_map), dtype=np.complex)
    for elem in value_map:
        if len(value_map[elem]) == 2:
            _x_complex[elem] = complex(*x_real[value_map[elem]])
        else:
            _x_complex[elem] = x_real[value_map[elem][0]]
    return _x_complex


__all__ = ['find_root']
