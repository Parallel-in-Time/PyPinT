# coding=utf-8
import numpy

from tests import NumpyAwareTestCase
from pypint.problems.i_problem import IProblem
from pypint.problems.transient_problem_mixin import TransientProblemMixin, problem_is_transient


class TransientProblemMixinTest(NumpyAwareTestCase):
    class ExampleProblem(IProblem, TransientProblemMixin):
        def __init__(self, *args, **kwargs):
            super(TransientProblemMixinTest.ExampleProblem, self).__init__(*args, **kwargs)
            TransientProblemMixin.__init__(self, *args, **kwargs)

        def print_lines_for_log(self):
            _lines = super(TransientProblemMixinTest.ExampleProblem, self).print_lines_for_log()
            _lines.update(TransientProblemMixin.print_lines_for_log(self))
            return _lines

        def __str__(self):
            _outstr = super(TransientProblemMixinTest.ExampleProblem, self).__str__()
            _outstr += TransientProblemMixin.__str__(self)
            return _outstr

    def setUp(self):
        self._default = TransientProblemMixinTest.ExampleProblem()

    def test_provides_time_start(self):
        self.assertEqual(self._default.time_start, 0.0)

        self._default.time_start = 0.5
        self.assertEqual(self._default.time_start, 0.5)

    def test_provides_time_end(self):
        self.assertEqual(self._default.time_end, 1.0)

        self._default.time_end = 2.0
        self.assertEqual(self._default.time_end, 2.0)

    def test_provides_time_interval(self):
        self.assertNumpyArrayEqual(self._default.time_interval, numpy.array([0.0, 1.0]))

        self._default.time_start = 1.0
        self.assertEqual(self._default.time_start, 1.0)
        self._default.time_end = 2.0
        self.assertEqual(self._default.time_end, 2.0)

        self._default.time_interval = numpy.array([0.5, 0.6])
        self.assertNumpyArrayEqual(self._default.time_interval, numpy.array([0.5, 0.6]))

        with self.assertRaises(ValueError):
            self._default.time_interval = numpy.array([0.8, 0.7])
        with self.assertRaises(ValueError):
            self._default.time_interval = numpy.array([0.8, 0.8])

    def test_constructor_provides_time_interval(self):
        _test_obj = TransientProblemMixinTest.ExampleProblem(time_start=1.0, time_end=2.0)
        self.assertEqual(_test_obj.time_start, 1.0)
        self.assertEqual(_test_obj.time_end, 2.0)

    def test_provides_time_interval_validation(self):
        self.assertTrue(self._default.validate_time_interval())

    def test_pretty_prints_time_interval(self):
        self.assertRegex(self._default.__str__(), "t in \[0.00, 1.00\]")

    def test_provides_lines_for_log(self):
        self.assertIn('Time Interval', self._default.print_lines_for_log())

    def test_transient_problem_inspection(self):
        self.assertTrue(problem_is_transient(self._default))


if __name__ == '__main__':
    import unittest
    unittest.main()
