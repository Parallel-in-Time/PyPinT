# coding=utf-8
"""Collection of assertions for validation.

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>

Examples
--------
>>> a = 3
>>> b = 2
>>> assert_condition(3 > 2, ValueError, "3 should be larger 2")

>>> func = lambda x: x
>>> assert_is_callable(func, "A lambda expression is callable.")

>>> l = [1, 2, 4]
>>> assert_is_in(2, l, "2 is in the list")

>>> m = {'hello': 'world'}
>>> assert_is_key(m, 'hello', "'hello' is a key")

>>> a = 1
>>> assert_is_instance(a, int, "'a' is an integer")
"""
import inspect
from collections import Callable

from pypint.utilities.tracing import checking_obj_name


def assert_condition(condition, exception_type, message, checking_obj=None):
    """Asserts trueness of arbitrary condition

    Parameters
    ----------
    condition : :py:class:`bool` or ``boolean expression``
        expression to be asserted

    exception_type : :py:class:`Exception`
        type of exception to be raised if ``condition`` evaluates to :py:class:`False`

    message : :py:class:`str`
        message content of exception raised

    checking_obj : :py:class:`object` or :py:class:`None`
        The exception will be raised in the scope of the given object.
        If :py:class:`None` (default) no scope will be displayed.

    Raises
    ------
    ``exception_type``
        if ``condition`` evaluates to :py:class:`False`
    """
    if not condition:
        raise exception_type("{:s}.{:s}(): {:s}"
                             .format(checking_obj_name(checking_obj), inspect.stack()[1][3], message))


def assert_is_callable(obj, message=None, checking_obj=None):
    """Asserts callability of given object

    Parameters
    ----------
    obj : :py:class:`object`
        object to be asserted callable

    message : :py:class:`str` or :py:class:`None`

    checking_obj : :py:class:`object` or :py:class:`None`
        The exception will be raised in the scope of the given object.
        If :py:class:`None` (default) no scope will be displayed.

    Raises
    ------
    ValueError
        if ``obj`` is not :py:class:`Callable`
    """
    if not isinstance(obj, Callable):
        if message is None:
            message = "Required a callable, received a '{:s}'.".format(type(obj))
        raise ValueError("{:s}.{:s}(): {:s}".format(checking_obj_name(checking_obj), inspect.stack()[2][3], message))


def assert_is_in(element, test_list, message, checking_obj=None):
    """Asserts element in list or sequence

    Parameters
    ----------
    element : :py:class:`object`
        element to check membership

    test_list : ``Sequence``
        sequence to check

    message : :py:class:`str`
        message content of exception raised

    checking_obj : :py:class:`object` or :py:class:`None`
        The exception will be raised in the scope of the given object.
        If :py:class:`None` (default) no scope will be displayed.

    Raises
    ------
    ValueError
        if ``element`` is not in ``test_list``
    """
    if element not in test_list:
        raise ValueError("{:s}.{:s}(): {:s}".format(checking_obj_name(checking_obj), inspect.stack()[2][3], message))


def assert_is_instance(obj, instances, message, checking_obj=None):
    """Asserts element is of certain type

    Parameters
    ----------
    obj : :py:class:`object`
        object to check type

    instances : :py:class:`type` of ``classes`` or :py:class:`class`
        types to check

    message : :py:class:`str`
        message content of exception raised

    checking_obj : :py:class:`object` or :py:class:`None`
        The exception will be raised in the scope of the given object.
        If :py:class:`None` (default) no scope will be displayed.

    Raises
    ------
    ValueError
        if ``obj`` is not of type ``instances``
    """
    if not isinstance(obj, instances):
        # make message optional and construct generic one by default
        raise ValueError("{:s}.{:s}(): {:s}".format(checking_obj_name(checking_obj), inspect.stack()[2][3], message))


def assert_is_key(dictionary, key, message, checking_obj=None):
    """Asserts dictionary has a certain key

    Parameters
    ----------
    dictionary : :py:class:`dict`
        dictionary to check

    key : ``key``
        key to check

    message : :py:class:`str`
        message content of exception raised

    checking_obj : :py:class:`object` or :py:class:`None`
        The exception will be raised in the scope of the given object.
        If :py:class:`None` (default) no scope will be displayed.

    Raises
    ------
    ValueError
        if ``key`` is not of a key in ``dictionary``
    """
    if not key in dictionary:
        raise ValueError("{:s}.{:s}(): {:s}".format(checking_obj_name(checking_obj), inspect.stack()[2][3], message))


__all__ = [
    'assert_condition',
    'assert_is_in',
    'assert_is_instance',
    'assert_is_callable',
    'assert_is_key'
]
