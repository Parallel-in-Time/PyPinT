# coding=utf-8
import unittest

from pypint.communicators.forward_sending_messaging import ForwardSendingMessaging
from pypint.communicators import Message


class Test(unittest.TestCase):
    def setUp(self):
        self._test_obj = ForwardSendingMessaging()
        self._prev = ForwardSendingMessaging()
        self._next = ForwardSendingMessaging()

    def test_solver_linking(self):
        self._test_obj.link_solvers(previous=self._prev, next=self._next)
        self.setUp()
        with self.assertRaises(ValueError):
            self._test_obj.link_solvers(previous=None, next=self._next)

    def test_sending(self):
        self._test_obj.link_solvers(previous=self._prev, next=self._next)
        _value = "a value"
        _flag = Message.SolverFlag.iterating
        self._test_obj.send(value=_value, flag=_flag)
        self.assertIs(self._next.buffer.value, _value)
        self.assertIsNone(self._next.buffer.time_point)
        self.assertIs(self._next.buffer.flag, _flag)

    def test_receiving(self):
        self.assertIsInstance(self._test_obj.receive(), Message)
        self.assertIsNone(self._test_obj.receive().value)
        self.assertIsNone(self._test_obj.receive().time_point)


if __name__ == '__main__':
    unittest.main()
