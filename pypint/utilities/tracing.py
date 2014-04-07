# coding=utf-8
"""Collection of utility functions related to the callstack and traceback.

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import inspect


def func_name(obj=None, *args, **kwargs):
    """Formats the calling functions name.

    Formats the calling functions name in the format ``'ClassName.FunctionName(): '``.

    Parameters
    ----------
    obj : :py:class:`object`
        Instance of an object the calling function is a member of.

    Returns
    -------
    formatted_function_name : :py:class:`str`
        Formatted function name of calling function.

    Examples
    --------
    >>> from pypint.utilities import func_name
    >>> class MyClass(object):
    ...     def my_func(self):
    ...         print(func_name(self) + "Hello World!")
    >>> my_obj = MyClass()
    >>> #my_obj.my_func()
    """
    _params = ''
    if len(args) > 0:
        _params += ', '.join(args)
        if len(kwargs) > 0:
            _params += ', '
    if len(kwargs) > 0:
        _c = 0
        for _k in kwargs:
            if _c > 0:
                _params += ', '
            _params += str(_k) + '=' + str(kwargs[_k])
            _c += 1

    if obj:
        return "%s<0x%x>.%s(%s): " % (checking_obj_name(obj), id(obj), inspect.stack()[1][3], _params)
    else:
        return "%s(%s): " % (inspect.stack()[1][3], _params)


def class_name(obj):
    return obj.__class__.__name__


def checking_obj_name(obj=None):
    return class_name(obj) if obj else "unknown"


__all__ = ['func_name', 'class_name']
