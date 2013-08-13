import unittest
from pySDC.integrate.quadrature import *

class QuadratureTest(unittest.TestCase):
    def testInitialization(self):
        with self.assertRaises(NotImplementedError):
            Quadrature()
    
def main():
    unittest.main()
    
if __name__ == "__main__":
    main()
