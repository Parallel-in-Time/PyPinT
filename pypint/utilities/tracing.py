# coding=utf-8
"""
Summary
-------
Collection of utility functions related to the callstack and traceback.

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""

import inspect


def func_name(obj=None):
    """
    Summary
    -------
    Formats the calling functions name.

    Extended Summary
    ----------------
    Formats the calling functions name in the format
    ``'ClassName.FunctionName(): '``.

    Parameters
    ----------
    obj : object
        Instance of an object the calling function is a member of.

    Returns
    -------
    formatted function name : str
        Formatted function name of calling function.

    Examples
    --------
    >>> from pypint.utilities import func_name
    >>> class MyClass(object):
    ...     def my_func(self):
    ...         print(func_name(self) + "Hello World!")
    >>> my_obj = MyClass()
    >>> my_obj.my_func()
    MyClass.my_func(): Hello World!
    """
    return "{:s}.{:s}(): "\
           .format(checking_obj_name(obj), inspect.stack()[1][3])


def assert_condition(condition, exception_type, message, checking_obj=None):
    if not condition:
        raise exception_type("{:s}.{:s}(): {:s}".format(checking_obj_name(checking_obj), inspect.stack()[2][3], message))


def assert_is_callable(obj, message=None, checking_obj=None):
    if not callable(obj):
        if message is None:
            message = "Required a callable, received a '{:s}'.".format(type(obj))
        raise ValueError("{:s}.{:s}(): {:s}".format(checking_obj_name(checking_obj), inspect.stack()[2][3], message))


def assert_is_instance(obj, instances, message, checking_obj=None):
    if not isinstance(obj, instances):
        # make message optional and construct generic one by default
        raise ValueError("{:s}.{:s}(): {:s}".format(checking_obj_name(checking_obj), inspect.stack()[2][3], message))


def assert_is_key(dictionary, key, message, checking_obj=None):
    if not key in dictionary:
        raise ValueError("{:s}.{:s}(): {:s}".format(checking_obj_name(checking_obj), inspect.stack()[2][3], message))


def checking_obj_name(obj=None):
    return obj.__class__.__name__ if obj is not None else "unknown"


__all__ = ['assert_is_callable', 'assert_is_instance', 'assert_condition', 'func_name']
