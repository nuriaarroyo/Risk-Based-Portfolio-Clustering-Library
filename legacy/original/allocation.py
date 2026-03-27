#papá de la recursión en asignacion de peso final hrp
from abc import ABC, abstractmethod
import pandas as pd 
import numpy as np 

class Allocation(ABC):
    @abstractmethod
    def weightall(self, cov: pd.DataFrame, orden: list):
        pass


class Naiverp(Allocation):
    def get_ivp(self, cov):
        #naive risk parity
        ivp = 1. / np.diag(cov.values)
        ivp /= ivp.sum()
        return ivp

    def get_cluster_var(self, cov, items):
        sub_cov = cov.loc[items, items]
        w = self.get_ivp(sub_cov)
        var = np.dot(w, np.dot(sub_cov, w))
        return var

    def weightall(self, cov: pd.DataFrame, orden: list) -> pd.Series:
        
        w = pd.Series(1.0, index=orden)
        clusters = [orden]

        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                # Dividir cluster en 2
                mid = len(cluster) // 2
                cluster1 = cluster[:mid]
                cluster2 = cluster[mid:]

                var1 = self.get_cluster_var(cov, cluster1)
                var2 = self.get_cluster_var(cov, cluster2)

                total_var = var1 + var2
                alloc1 = 1 - var1 / total_var
                alloc2 = 1 - alloc1

                w[cluster1] *= alloc1
                w[cluster2] *= alloc2

                new_clusters += [cluster1, cluster2]

            clusters = new_clusters

        return w
