from decimal import Decimal
from pySDC.integrate.quadrature import Quadrature

class NewtonCotes( Quadrature ):
    """
    """

    def __init__( self ):
        """
        """

    @staticmethod
    def integrate( func=lambda x: 1, begin=Decimal( 0 ), end=Decimal( 1 ), steps=10, order=1 ):
        """
        """
        a = Decimal( begin )
        b = Decimal( end )

        if a == b or ( b - a ) <= Decimal( 0.0 ):
            raise AttributeError( "Integration interval must be non-zero positive (end - begin = " + str( b - a ) + ")." )
        if steps < 1:
            raise AttributeError( "At least one step makes sense (steps=" + str( steps ) + ")." )

        step_width = ( b - a ) / Decimal( steps )
        result = Decimal( 0.0 )

        if order == 1:
            # Midpoint rule
            for i in range( 0, steps ):
                result += step_width * Decimal( func( a + Decimal( i + 0.5 ) * step_width ) )
        elif order == 2:
            # Trapezoid rule
            for i in range( 0, steps ):
                result += step_width * Decimal( 
                                                  func( a + i * step_width )
                                                + func( a + ( i + 1 ) * step_width )
                                              ) / Decimal( 2 )
        elif order == 3:
            # Simpson rule
            for i in range( 0, steps ):
                result += step_width * Decimal( 
                                                  func( a + i * step_width )
                                                + 4 * func( a + Decimal( i + 0.5 ) * step_width )
                                                + func( a + Decimal( i + 1 ) * step_width )
                                              ) / Decimal( 6 )
        elif order == 4:
            # Simpson 3/8 rule
            for i in range( 0, steps ):
                result += step_width * Decimal( 
                                                   func( a + i * step_width )
                                                 + 3 * func( a + Decimal( i + 1 / Decimal( 3 ) ) * step_width )
                                                 + 3 * func( a + Decimal( i + 2 / Decimal( 3 ) ) * step_width )
                                                 + func( a + Decimal( i + 1 ) * step_width )
                                               ) / Decimal( 8 )

        else:
            raise NotImplementedError( "Newton-Codes integration scheme with order=" + str( order ) + " not implemented." )

        return result
