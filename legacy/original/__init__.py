"""
Librería de Portafolios de Inversión

Esta librería implementa diferentes estrategias de construcción de portafolios:
- Naive (1/N)
- Markowitz (Optimización de Sharpe)
- HRP (Hierarchical Risk Parity)

Autor: Nuria Arroyo Bustamante 
"""

from .portafolio import Portafolio
from .naive import NaivePortafolio
from .markowitz import MarkowitzPortafolio
from .hrp_style import HRPStyle
from .distance import Distancia, distdecorr
from .clustering import Clustering, SingleLinkage
from .allocation import Allocation, Naiverp

__version__ = "1.0.0"
__author__ = "Nuria Arroyo"

__all__ = [
    'Portafolio',
    'NaivePortafolio', 
    'MarkowitzPortafolio',
    'HRPStyle',
    'Distancia',
    'distdecorr',
    'Clustering',
    'SingleLinkage',
    'Allocation',
    'Naiverp'
]
