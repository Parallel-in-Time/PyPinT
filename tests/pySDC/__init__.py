import unittest
from tests.pySDC.integrate.__init__ import IntegrateTests

class pySDCTests( unittest.TestSuite ):
    def __init__( self ):
        self.addTests( IntegrateTests() )

if __name__ == "__main__":
    unittest.main()
