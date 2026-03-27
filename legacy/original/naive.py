# la primogenita 
import numpy as np 
import pandas as pd

from original.portafolio import Portafolio 


class NaivePortafolio(Portafolio):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data, "Portafolio Naive (1/N)")

    def construir(self,
                  construct_start: str, construct_end: str,
                  bt_train_start: str, bt_train_end: str,
                  bt_test_start: str, bt_test_end: str):
        """
        Construye un portafolio 1/N usando SIEMPRE el tramo de construcción (triple split).
        - define los tres tramos con `dividir(...)`
        - fija pesos equiponderados sobre el universo de construcción
        - construye el portafolio (serie de retornos en construcción)

        Returns
        -------
        np.ndarray
            Vector de pesos (ordenados según `self._construct_tickers`).
        """
        # 1) triple split
        self.dividir(
            construct_start, construct_end,
            bt_train_start, bt_train_end,
            bt_test_start, bt_test_end
        )

        # 2) universo de construcción y pesos 1/N
        n = len(self._construct_tickers or [])
        if n == 0:
            raise RuntimeError("No hay activos en el tramo de construcción. Revisa tus fechas.")
        pesos = np.full(n, 1.0 / n, dtype=float)

        # 3) construir SOLO sobre el set de construcción
        super().construir(pesos)

        return pesos
