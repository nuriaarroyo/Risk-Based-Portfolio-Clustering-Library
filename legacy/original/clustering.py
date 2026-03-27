# papá de los métodos de clustering 
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd 


class Clustering(ABC):
    @abstractmethod
    def cluster(self, distancia: np.ndarray):
        pass

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

class SingleLinkage(Clustering):
    def cluster(self, distancia: pd.DataFrame) -> np.ndarray:
        condensed = squareform(distancia.values, checks=False)
        return linkage(condensed, method='single')