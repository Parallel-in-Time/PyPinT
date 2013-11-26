# coding=utf-8

from pypint.communicators.i_communication_provider import ICommunicationProvider
import unittest


class ICommunicationProviderTest(unittest.TestCase):
    def test_initialization(self):
        _test_obj = ICommunicationProvider()
