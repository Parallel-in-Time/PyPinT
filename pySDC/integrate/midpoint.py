from decimal import Decimal
from pySDC.integrate.newton_cotes import NewtonCotes

class Midpoint( NewtonCotes ):
	"""
	"""

	def __init__( self ):
		"""
		"""

	@staticmethod
	def integrate( func=lambda x: 1, begin=Decimal( 0 ), end=Decimal( 1 ), steps=10 ):
		"""
		"""
		a = Decimal( begin )
		b = Decimal( end )
		
		if a == b or ( b - a ) <= Decimal( 0.0 ):
			raise AttributeError( "Integration interval must be non-zero positive (end - begin = " + str( b - a ) + ")." )
		if steps < 1:
			raise AttributeError( "At least one step makes sense (steps=" + str(steps) + ")." )

		w = ( b - a ) / Decimal( steps )
		result = Decimal( 0.0 )
		for i in range( 0, steps ):
			result += w * Decimal( func( a - w / 2 + i * w ) )
		return result
