# coding=utf-8
from tests import NumpyAwareTestCase
import numpy
from pypint.solvers.diagnosis.norms import supremum_norm, two_norm
from pypint.solvers.diagnosis import IDiagnosisValue


class NormsTest(NumpyAwareTestCase):
    def setUp(self):
        self._value = IDiagnosisValue(numpy.array([1.0, 2.0, 3.0]))

    def test_supremums_norm(self):
        self.assertEqual(supremum_norm(self._value), 3.0)
        self.assertEqual(supremum_norm(self._value.value), 3.0)

    def test_two_norm(self):
        self.assertEqual(two_norm(self._value), 3.7416573867739413)
        self.assertEqual(two_norm(self._value.value), 3.7416573867739413)


if __name__ == '__main__':
    import unittest
    unittest.main()
