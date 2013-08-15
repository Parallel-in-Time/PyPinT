from decimal import Decimal
from pySDC.integrate.quadrature import Quadrature

class Gauss( Quadrature ):
    """
    """

    def __init__( self ):
        """
        """

    @staticmethod
    def integrate( func=lambda x: Decimal( 1.0 ), begin=0, end=1, nPoints=3 ):
        """
        """
        a = Decimal( begin )
        b = Decimal( end )

        if a == b or ( b - a ) <= Decimal( 0.0 ):
            raise ValueError( "Integration interval must be non-zero positive (end - begin = " + str( b - a ) + ")." )

        transform = [ Decimal( ( b - a ) / Decimal( 2 ) ), Decimal( ( b + a ) / Decimal( 2 ) ) ]
        pointsAndWeights = Gauss._pointsAndWeights( nPoints )
        result = Decimal( 0.0 )
        for pointWeight in pointsAndWeights:
            result += pointWeight[1] * Decimal( func( transform[0] * pointWeight[0] + transform[1] ) )
        result *= transform[0]
        return result

    @staticmethod
    def _pointsAndWeights( nPoints ):
        if nPoints == 3:
            return [ [ Decimal( -1.0 ), Decimal( 1 / Decimal( 3 ) )],
                     [ Decimal( 0.0 ), Decimal( 4 / Decimal( 3 ) )],
                     [Decimal( 1.0 ), Decimal( 1 / Decimal( 3 ) )]
                   ]
        elif nPoints == 4:
            return [ [ Decimal( -1.0 ), Decimal( 1 / Decimal( 6 ) ) ],
                     [ Decimal( -1.0 ) / Decimal( 5 ) * Decimal( 5 ).sqrt(), Decimal( 5 ) / Decimal( 6 ) ],
                     [ Decimal( 1.0 ) / Decimal( 5 ) * Decimal( 5 ).sqrt(), Decimal( 5 ) / Decimal( 6 ) ],
                     [ Decimal( 1.0 ), Decimal( 1 / Decimal( 6 ) ) ]
                   ]
        elif nPoints < 3:
            raise ValueError( "Gauss-Lobatto quadrature does not work with less than three points." )
        else:
            raise NotImplementedError()
