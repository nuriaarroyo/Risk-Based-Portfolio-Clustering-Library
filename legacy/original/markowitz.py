from  scipy.optimize import minimize
import numpy as np 
import pandas as pd

from original.portafolio import Portafolio 


class MarkowitzPortafolio(Portafolio):
    def __init__(self, data):
        super().__init__(data, "Portafolio Markowitz")

    def construir(self, start_train_date, end_train_date, start_bt_date, end_bt_date):
        # Dividir los datos y calcular retornos esperados y volatilidad
        self.dividir(start_train_date, end_train_date, start_bt_date, end_bt_date)
        
        # Usar los datos de entrenamiento ya divididos
        closeprices = self.data_train.xs('Close', axis=1, level=1)

        # Calcular retornos logarítmicos
        logreturns = np.log(closeprices / closeprices.shift(1)).dropna()
        mean_returns = logreturns.mean()
        cov_matrix = logreturns.cov()
        
        # Obtener número de activos
        n = len(self.data_train.columns.get_level_values(0).unique())

        # Restricciones: pesos suman 1
        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
        bounds = tuple((0, 1) for _ in range(n))
        initial_weights = np.ones(n) / n 

        # Optimización para maximizar ratio de Sharpe
        result = minimize(
            fun=self._negative_sharpe_ratio,
            x0=initial_weights,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            self.weights = result.x
            self.mean_returns = mean_returns
            self.cov_matrix = cov_matrix
            
            # Asignar los pesos al portafolio
            super().construir(self.weights)
            
            return self.weights
        else:
            print("Optimización fallida:", result.message)
            return None

    @staticmethod
    def _portfolio_return(weights, mean_returns):
        return weights.T @ mean_returns

    @staticmethod
    def _portfolio_variance(weights, cov_matrix):
        return weights.T @ cov_matrix @ weights

    def _negative_sharpe_ratio(self, weights, mean_returns, cov_matrix):
        port_ret = self._portfolio_return(weights, mean_returns)
        port_var = self._portfolio_variance(weights, cov_matrix)
        port_vol = np.sqrt(port_var)
        return -port_ret / port_vol  