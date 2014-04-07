# coding=utf-8
import unittest

from pypint.communicators.i_communication_provider import ICommunicationProvider
from pypint.communicators import Message


class ICommunicationProviderTest(unittest.TestCase):
    def setUp(self):
        self._test_obj = ICommunicationProvider()

    def test_allows_custom_buffer_location(self):
        _buffer = Message()
        _buffer.value = "a buffer"
        _buffer.time_point = 0.1
        _with_buffer = ICommunicationProvider(buffer=_buffer)
        self.assertIs(_with_buffer.buffer, _buffer)

    def test_provides_interface_for_sending(self):
        self._test_obj.send()

    def test_provides_interface_for_receiving(self):
        self._test_obj.receive()

    def test_provides_interface_for_linking(self):
        self._test_obj.link_solvers()

    def test_provides_interface_for_writing_into_buffer(self):
        _value = "a value"
        _time_point = 0.1
        _flag = Message.SolverFlag.iterating

        self._test_obj.write_buffer(value=_value)
        self.assertIs(self._test_obj.buffer.value, _value)
        self.assertIsNone(self._test_obj.buffer.time_point)
        self.assertIs(self._test_obj.buffer.flag, Message.SolverFlag.none)

        self._test_obj.write_buffer(time_point=_time_point)
        self.assertEqual(self._test_obj.buffer.time_point, _time_point)
        self.assertIs(self._test_obj.buffer.value, _value)
        self.assertIs(self._test_obj.buffer.flag, Message.SolverFlag.none)

        self._test_obj.write_buffer(flag=_flag)
        self.assertIs(self._test_obj.buffer.flag, _flag)

        self.setUp()
        self._test_obj.write_buffer(value=_value, flag=_flag)
        self.assertIs(self._test_obj.buffer.value, _value)
        self.assertIsNone(self._test_obj.buffer.time_point)
        self.assertIs(self._test_obj.buffer.flag, _flag)


if __name__ == '__main__':
    unittest.main()
