#papa de las distancias 

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class Distancia(ABC):
    @abstractmethod
    def compute(self, returns):
     pass

class distdecorr(Distancia):
    def compute(self, returns):
        corr = returns.corr()
        dist = np.sqrt((1 - corr) / 2.0)
        return dist
    
    