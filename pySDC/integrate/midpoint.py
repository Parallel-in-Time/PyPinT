from decimal import Decimal
from pySDC.integrate.quadrature import Quadrature

class Midpoint( Quadrature ):
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
		w = ( b - a ) / Decimal( steps )
		result = Decimal( 0.0 )
		for i in range( 0, steps ):
			result += w * Decimal( func( a - w / 2 + i * w ) )
		return result
