# coding=utf-8
"""
Communicators on top of Iterative Time Solvers

.. moduleauthor:: Torbjörn Klatt <t.klatt@fz-juelich.de>
.. moduleauthor:: Dieter Moser <d.moser@fz-juelich.de>
"""

from .i_communication_provider import *
from .i_linear_messaging import *

__all__ = ['ICommunicationProvider', 'ILinearMessaging']
