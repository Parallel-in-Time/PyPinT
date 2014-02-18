# coding=utf-8
"""Collection of assertions for validation.

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import inspect
from collections import Callable

from pypint.utilities.tracing import checking_obj_name


def assert_condition(condition, exception_type, message, checking_obj=None):
    if not condition:
        raise exception_type("{:s}.{:s}(): {:s}"
                             .format(checking_obj_name(checking_obj), inspect.stack()[1][3], message))


def assert_is_callable(obj, message=None, checking_obj=None):
    if not isinstance(obj, Callable):
        if message is None:
            message = "Required a callable, received a '{:s}'.".format(type(obj))
        raise ValueError("{:s}.{:s}(): {:s}".format(checking_obj_name(checking_obj), inspect.stack()[2][3], message))


def assert_is_in(element, test_list, message, checking_obj=None):
    if element not in test_list:
        raise ValueError("{:s}.{:s}(): {:s}".format(checking_obj_name(checking_obj), inspect.stack()[2][3], message))


def assert_is_instance(obj, instances, message, checking_obj=None):
    if not isinstance(obj, instances):
        # make message optional and construct generic one by default
        raise ValueError("{:s}.{:s}(): {:s}".format(checking_obj_name(checking_obj), inspect.stack()[2][3], message))


def assert_is_key(dictionary, key, message, checking_obj=None):
    if not key in dictionary:
        raise ValueError("{:s}.{:s}(): {:s}".format(checking_obj_name(checking_obj), inspect.stack()[2][3], message))


__all__ = [
    'assert_condition',
    'assert_is_in',
    'assert_is_instance',
    'assert_is_callable',
    'assert_is_key'
]
