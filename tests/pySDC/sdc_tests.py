import unittest
from pySDC.sdc import SDC


class SDCTests(unittest.TestCase):
    def test_sdc_initialization(self):
        SDC()

    def test_sdc_default_values(self):
        my_sdc = SDC()
        #self.assertEqual(my_sdc.initial_value , 0.0, "default initial value")
        #self.assertEqual(my_sdc.timeSteps, 10, "default time steps")
        #self.assertEqual(my_sdc.numSubsteps, 3, "default number of substeps")
        #self.assertEqual(my_sdc.iterations, 5, "default number if SDC iterations")
        my_sdc.solve()
        my_sdc.print_solution()


if __name__ == "__main__":
    unittest.main()
