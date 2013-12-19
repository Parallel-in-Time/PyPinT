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
    obj_name = obj.__class__.__name__ if obj is not None else "unknown"
    return "{:s}.{:s}(): "\
           .format(obj_name, inspect.stack()[1][3])


def critical_assert(condition, exception_type, message, checking_obj=None):
    if not condition:
        checking_obj_name = checking_obj.__class__.__name__ if checking_obj is not None else "unknown"
        raise exception_type("{:s}.{:s}(): {:s}".format(checking_obj_name, inspect.stack()[2][3], message))


__all__ = ['func_name', 'critical_assert']
