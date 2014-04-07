# coding=utf-8
import unittest

import numpy

from pypint.integrators.sdc_integrator import SdcIntegrator
from tests.__init__ import NumpyAwareTestCase


class SdcIntegratorTest(NumpyAwareTestCase):
    def setUp(self):
        self._test_obj = SdcIntegrator()

    def test_s_and_q_matrix_computation_default_interval(self):
        self._test_obj.init(num_nodes=3)
        computed_smat = self._test_obj._smat
        expected_smat = numpy.array(
            [
                [0.416666666666667, 0.666666666666667, -0.0833333333333335],
                [-0.0833333333333333, 0.666666666666667, 0.416666666666667]
            ]
        )
        self.assertNumpyArrayAlmostEqual(computed_smat, expected_smat, delta=1e-8)

    def test_q_matrix_computation_default_interval(self):
        self._test_obj.init(num_nodes=3)
        computed_qmat = self._test_obj._qmat
        expected_qmat = numpy.array(
            [
                [0.0, 0.0, 0.0],
                [0.416666666666667, 0.666666666666667, -0.0833333333333335],
                [float(1.0/3.0), float(4.0/3.0), float(1.0/3.0)]
            ]
        )
        self.assertNumpyArrayAlmostEqual(computed_qmat, expected_qmat, delta=1e-8)

    def test_s_matrix_computation_0_to_1_interval(self):
        self._test_obj.init(num_nodes=3, interval=numpy.array([0.0, 1.0]))
        computed_smat = self._test_obj._smat
        expected_smat = numpy.array(
            [
                [0.208333333333333, 0.333333333333334, -0.0416666666666667],
                [-0.0416666666666667, 0.333333333333333, 0.208333333333333]
            ]
        )
        self.assertNumpyArrayAlmostEqual(computed_smat, expected_smat, delta=1e-8)

    def test_q_matrix_computation_0_to_1_interval(self):
        self._test_obj.init(num_nodes=3, interval=numpy.array([0.0, 1.0]))
        computed_qmat = self._test_obj._qmat
        expected_qmat = numpy.array(
            [
                [0.0, 0.0, 0.0],
                [0.208333333333333, 0.333333333333334, -0.0416666666666667],
                [0.166666666666667, 0.666666666666667, 0.1666666666666667]
            ]
        )
        self.assertNumpyArrayAlmostEqual(computed_qmat, expected_qmat, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
