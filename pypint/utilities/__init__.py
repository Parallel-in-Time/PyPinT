# coding=utf-8
"""
Summary
-------
Collection of various utility functions.

.. moduleauthor: 'Torbjörn Klatt' <t.klatt@fz-juelich.de>
"""

from .tracing import assert_is_callable, assert_is_instance, critical_assert, func_name

__all__ = ['assert_is_callable', 'assert_is_instance', 'critical_assert', 'func_name']
