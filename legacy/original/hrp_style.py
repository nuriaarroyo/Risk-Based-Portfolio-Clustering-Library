#hija de portafolio con 3 main inputs 

import numpy as np 
import pandas as pd 

from original.portafolio import Portafolio

class HRPStyle(Portafolio):
    def __init__(self, data, distancia, clustering, allocation):
        super().__init__(data, "Portafolio HRP")
        self.distancia = distancia 
        self.clustering = clustering
        self.allocation = allocation

    def quasid(self, link):
        """
        Método equivalente a getQuasiDiag: ordena los activos a partir del linkage.
        """
        link = link.astype(int)
        sortIx = pd.Series([link[-1, 0], link[-1, 1]])
        numItems = link[-1, 3]

        while sortIx.max() >= numItems:
            sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # duplicamos los espacios
            df0 = sortIx[sortIx >= numItems]  # buscamos los que aún son clusters
            i = df0.index
            j = df0.values - numItems  # ajustamos índice
            sortIx[i] = link[j, 0]
            df1 = pd.Series(link[j, 1], index=i + 1)
            sortIx = pd.concat([sortIx, df1])  # Usar concat en lugar de append
            sortIx = sortIx.sort_index()
            sortIx.index = range(sortIx.shape[0])  # reindexar limpio

        return sortIx.tolist()

    def construir_hrp(self, start_train_date, end_train_date, start_bt_date, end_bt_date):
        # Dividir los datos y calcular retornos esperados y volatilidad
        self.dividir(start_train_date, end_train_date, start_bt_date, end_bt_date)
        
        # Usar la property retornos del período de entrenamiento
        retornos = self.data_train.xs('Close', axis=1, level=1).pct_change().dropna()
        cov = retornos.cov()

        # Calcular matriz de distancia
        distmat = self.distancia.compute(retornos)
        
        # Aplicar clustering
        linkmat = self.clustering.cluster(distmat)
        
        # Obtener orden cuasidiagonal
        order = self.quasid(linkmat)

        # Convertir índices a nombres reales si son enteros
        if isinstance(order[0], int):
            tickers = self.data_train.columns.get_level_values(0).unique().to_list()
            order = [tickers[i] for i in order]

        # Asignar pesos usando el método de allocation
        pesos_series = self.allocation.weightall(cov, order)
        
        # Obtener pesos en el orden correcto de los tickers
        tickers = self.data_train.columns.get_level_values(0).unique().to_list()
        pesos = pesos_series.loc[tickers].values

        # Asignar los pesos al portafolio
        super().construir(pesos)
        
        return pesos
