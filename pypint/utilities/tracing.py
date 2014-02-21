# coding=utf-8
"""Collection of utility functions related to the callstack and traceback.

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
"""
import inspect

#TODO:  Form des inputs und outputs einer funktion testen, waere sicher praktisch
def func_input(func,input_form):
    pass
def func_output(func,output_form):
    pass


def func_name(obj=None):
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
    >>> my_obj.my_func()
    MyClass.my_func(): Hello World!
    """
    return "{:s}.{:s}(): "\
           .format(checking_obj_name(obj), inspect.stack()[1][3])


def checking_obj_name(obj=None):
    return obj.__class__.__name__ if obj is not None else "unknown"


__all__ = ['func_name']
