# coding=utf-8

from pypint.communicators.i_linear_messaging import ILinearMessaging
import unittest


class ILinearMessagingTest(unittest.TestCase):
    def test_initialization(self):
        _test_obj = ILinearMessaging()
