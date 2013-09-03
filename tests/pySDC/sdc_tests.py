import unittest
from decimal import Decimal
from pySDC.sdc import SDC

class SDCTests(unittest.TestCase):
    def test_initialization(self):
        SDC()

    def test_default_values(self):
        mySDC = SDC()
        self.assertEqual(mySDC.initial_value , Decimal(0.0), "default initial value")
        self.assertEqual(mySDC.timeSteps, 10, "default time steps")
        self.assertEqual(mySDC.numSubsteps, 3, "default number of substeps")
        self.assertEqual(mySDC.iterations, 5, "default number if SDC iterations")
        mySDC.solve()

if __name__ == "__main__":
    unittest.main()
