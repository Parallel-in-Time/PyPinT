# coding=utf-8

import unittest
from tests.__init__ import NumpyAwareTestCase
from pypint.plugins.implicit_solvers.find_root import find_root, _transform_to_real, _transform_to_complex
import numpy as np


class FindRootTest(NumpyAwareTestCase):
    def testTransformFromComplex(self):
        _complex = np.array([1 + 1j], dtype=np.complex)
        _real = _transform_to_real(_complex, {0: [0, 1]}, 2)
        self.assertNumpyArrayAlmostEqual(_real, np.array([1.0, 1.0], dtype=np.float))

    def testTransformToComplex(self):
        _real = np.array([1.0, 1.0], dtype=np.float)
        _complex = _transform_to_complex(_real, {0: [0, 1]})
        self.assertNumpyArrayAlmostEqual(_complex, np.array([1+1j], dtype=np.complex))

    def testSimpleRoot(self):
        _func = lambda x: np.array([-1.0 + 1.0j, -1.0], dtype=np.complex) + x
        _in_x = np.array([0.0, 0.0], dtype=np.complex)
        _sol = find_root(_func, _in_x)
        self.assertTrue(_sol.success)
        self.assertNumpyArrayAlmostEqual(_sol.x, np.array([1.0 - 1.0j, 1.0], dtype=np.complex))

if __name__ == "__main__":
    unittest.main()
