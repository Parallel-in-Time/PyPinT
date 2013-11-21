# coding=utf-8

from pypint.integrators.sdc_integrator import SdcIntegrator
import unittest
import numpy
from tests.__init__ import NumpyAwareTestCase


class SdcIntegratorTest(NumpyAwareTestCase):
    def setUp(self):
        self._test_obj = SdcIntegrator()

    def test_initialization(self):
        self._test_obj.init()

    def test_s_matrix_computation_default_interval(self):
        self._test_obj.init(num_nodes=3)
        computed_smat = self._test_obj._smat
        expected_smat = numpy.array(
            [
                [0.416666666666667, 0.666666666666667, -0.0833333333333335],
                [-0.0833333333333333, 0.666666666666667, 0.416666666666667]
            ]
        )
        self.assertNumpyArrayAlmostEqual(computed_smat, expected_smat, delta=1e-8)

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


if __name__ == "__main__":
    unittest.main()
