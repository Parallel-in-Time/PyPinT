# coding=utf-8
import numpy

from tests import NumpyAwareTestCase
from pypint.solvers.diagnosis.i_diagnosis_value import IDiagnosisValue


class IDiagnosisTest(NumpyAwareTestCase):
    def setUp(self):
        self._value = numpy.array([1, 2, 3])
        self._default = IDiagnosisValue(self._value)

    def test_wrapps_a_value(self):
        self.assertNumpyArrayEqual(self._default.value, self._value)

    def test_knows_value_data_type(self):
        self.assertEqual(self._default.numeric_type, numpy.int)

    def test_is_equality_comparable(self):
        _test = IDiagnosisValue(self._value)
        self.assertEqual(self._default, _test)
        self.assertTrue(self._default == _test)
        self.assertTrue(self._default.__eq__(_test))
        self.assertFalse(self._default != _test)
        self.assertFalse(self._default.__ne__(_test))

    def test_is_not_orderable(self):
        _test = IDiagnosisValue(self._value)

        with self.assertRaises(TypeError):
            self._default < _test
        self.assertEqual(self._default.__lt__(_test), NotImplemented)

        with self.assertRaises(TypeError):
            self._default <= _test
        self.assertEqual(self._default.__le__(_test), NotImplemented)

        with self.assertRaises(TypeError):
            self._default > _test
        self.assertEqual(self._default.__gt__(_test), NotImplemented)

        with self.assertRaises(TypeError):
            self._default >= _test
        self.assertEqual(self._default.__ge__(_test), NotImplemented)


if __name__ == '__main__':
    import unittest
    unittest.main()
