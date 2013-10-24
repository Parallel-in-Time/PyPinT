"""
Quadrature Interface
"""


class Quadrature(object):
    """
    Provides interface for quadrature.
    """

    def __init__(self):
        """
        Initialization
        """
        pass

    @staticmethod
    def integrate():
        """
        Interface for integration
        """
        raise NotImplementedError("Should be implemented by specific scheme.")
