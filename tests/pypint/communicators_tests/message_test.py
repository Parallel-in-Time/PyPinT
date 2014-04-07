# coding=utf-8
import unittest

from pypint.communicators.message import Message


class MessageTest(unittest.TestCase):
    def setUp(self):
        self._test_obj = Message()

    def test_provides_solver_state_flags(self):
        _none = Message.SolverFlag.none
        _iterating = Message.SolverFlag.iterating
        _converged = Message.SolverFlag.converged
        _finished = Message.SolverFlag.finished
        _failed = Message.SolverFlag.failed
        _time_adjusted = Message.SolverFlag.time_adjusted

    def test_has_a_value(self):
        self.assertIsNone(self._test_obj.value)
        self._test_obj.value = "some not further specified value"
        self.assertIs(self._test_obj.value, "some not further specified value")
        del self._test_obj.value
        self.assertIsNone(self._test_obj.value)

    def test_provides_a_time_point(self):
        self.assertIsNone(self._test_obj.time_point)
        self._test_obj.time_point = 0.1
        self.assertEqual(self._test_obj.time_point, 0.1)
        del self._test_obj.time_point
        self.assertIsNone(self._test_obj.time_point)

    def test_has_a_solver_flag(self):
        self.assertIs(self._test_obj.flag, Message.SolverFlag.none)
        self._test_obj.flag = Message.SolverFlag.iterating
        self.assertIs(self._test_obj.flag, Message.SolverFlag.iterating)
        with self.assertRaises(ValueError):
            self._test_obj.flag = "not a valid flag"
        del self._test_obj.flag
        self.assertIs(self._test_obj.flag, Message.SolverFlag.none)


if __name__ == '__main__':
    unittest.main()
