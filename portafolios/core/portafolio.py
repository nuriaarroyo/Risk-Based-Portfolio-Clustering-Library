# portafolios/core/portafolio.py
from __future__ import annotations
import pandas as pd
from typing import Iterable, Optional, Callable, Any
from ..metrics import asset as am
from ..metrics import portfolio as pm
from ..plots import pie,bar,bubble,corr_heatmap



class Portfolio:
    """
    Una vez mas horrores by NAB
    """

    def __init__(
        self,
        *,
        tickers: Optional[Iterable[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        loader: Optional[Callable[..., pd.DataFrame]] = None,
        loader_kwargs: Optional[dict[str, Any]] = None,
        freq: Optional[str] = "B"):
        #para guaradar
        #general del objeto
        self.tickers = list(tickers) if tickers is not None else None
        self.start = pd.Timestamp(start) if start else None
        self.end = pd.Timestamp(end) if end else None
        self.freq = freq
        self.loader = loader
        self.loader_kwargs = loader_kwargs or {}
        self.prices: Optional[pd.DataFrame] = None  # SOLO close prices en esta fase
        # de assets
        self.asset_returns = None
        self.asset_log_returns = None
        self.weights = None
        self.asset_vol = None
        self.asset_vol_lr = None
        self.asset_mean_ret =None
        self.asset_mean_lr = None
        # de portafolio
        self.covariance = None
        self.correlation = None
        #aca agregar las que vengand de metrics portafolio 

        
        self.info = {}

    def preparar_datos(self) -> "Portfolio":
        """
        Llama al loader y pobla:
          - self.prices: DataFrame de precios de cierre por ticker
          - self.tickers: lista final de tickers (si no se pasó al inicio)
        """
        if self.loader is None:
            raise ValueError("Debes proporcionar un 'loader' que devuelva precios de cierre.")

        # El loader espera argumentos por nombre
        df = self.loader(
            tickers=self.tickers,
            start=(self.start.isoformat() if self.start is not None else None),
            end=(self.end.isoformat() if self.end is not None else None),
            freq=self.freq,
            **self.loader_kwargs
        )

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("El loader debe devolver un DataFrame con DatetimeIndex.")
        if df.empty:
            raise ValueError("El loader devolvió un DataFrame vacío.")

        self.prices = df.sort_index()
        # Si no definiste tickers al crear el objeto, toma los del DF
        if self.tickers is None:
            self.tickers = list(self.prices.columns)

        self.asset_returns     = am.returns_simple(self.prices)
        self.asset_log_returns = am.returns_log(self.prices)
        self.asset_vol         = am.volatility(self.asset_returns)
        self.asset_vol_lr        = am.volatility(self.asset_log_returns)
        self.asset_mean_ret = am.mean_return(self.asset_returns)
        self.asset_mean_lr = am.mean_return(self.asset_log_returns)
        self.covariance = am.covariance_matrix(self.asset_log_returns)
        self.correlation = am.correlation_matrix(self.asset_log_returns)
        

        return self
    
        # portafolios/core/portafolio.py (fragmento de construir)


    def construir(self, constructor,**kwargs) -> "Portfolio":
        if self.asset_returns is None:
            raise RuntimeError("Llama primero a preparar_datos().")
        
        w, meta = constructor.optimizar(self.asset_returns, **kwargs)
        self.weights = w.reindex(self.asset_returns.columns).fillna(0).sort_index()

        self.info.update({
            "constructor": getattr(constructor, "nombre", str(constructor)),
            **(meta or {})
            
        #aca calcularlas como hice en proparar datos para las de assets y lo que son las entradas de contruccion

        })
        return self


 

    def kpi(self, nombre: str, **kwargs):
        """
        Router de KPIs de portafolio.
        Soporta (por momentos): 'exp_return', 'vol', 'sharpe_m', 'rc'
        Por trayectoria: 'sharpe', 'sortino', 'mdd', 'calmar', 'var', 'cvar', 'te', 'alpha_beta', 'ir'
        """
        if self.weights is None:
            raise RuntimeError("Primero construye el portafolio para tener weights.")

        ann = kwargs.get("ann_factor", None)
        rf  = kwargs.get("rf_per_period", 0.0)

        # --- KPIs por momentos (rápidos y estables)
        if nombre in ("exp_return", "er"):
            return pm.expected_return_from_moments(self.asset_mean_ret, self.weights, ann_factor=ann)
        if nombre in ("vol", "volatility"):
            return pm.expected_volatility_from_moments(self.covariance, self.weights, ann_factor=ann)
        if nombre in ("sharpe_m", "sharpe_moments"):
            return pm.sharpe_from_moments(self.asset_mean_ret, self.covariance, self.weights,
                                        rf_per_period=rf, ann_factor=ann)
        if nombre in ("rc", "risk_contrib"):
            return pm.risk_contributions_from_cov(self.covariance, self.weights,
                                                ann_factor=ann, as_fraction=kwargs.get("as_fraction", True))

        # --- KPIs por trayectoria (requieren serie de retornos del portafolio)
        if nombre in ("sharpe",):
            return pm.sharpe(self.asset_returns, self.weights, rf_per_period=rf, ann_factor=ann)
        if nombre in ("sortino",):
            return pm.sortino(self.asset_returns, self.weights,
                            mar_per_period=kwargs.get("mar_per_period", 0.0), ann_factor=ann)
        if nombre in ("mdd", "max_drawdown"):
            return pm.max_drawdown(self.asset_returns, self.weights)
        if nombre in ("calmar",):
            return pm.calmar_from_moments(self.asset_mean_ret, self.covariance, self.weights,
                                        ann_factor=ann, returns_for_mdd=self.asset_returns)
        if nombre in ("var", "var_gauss"):
            return pm.var_gaussian(self.asset_returns, self.weights, alpha=kwargs.get("alpha", 0.95), ann_factor=ann)
        if nombre in ("cvar", "cvar_gauss", "es"):
            return pm.cvar_gaussian(self.asset_returns, self.weights, alpha=kwargs.get("alpha", 0.95), ann_factor=ann)
        if nombre in ("te", "tracking_error"):
            bench = kwargs["benchmark"]  # Serie de retornos del benchmark
            return pm.tracking_error(self.asset_returns, self.weights, bench, ann_factor=ann)
        if nombre in ("alpha_beta", "ab"):
            bench = kwargs["benchmark"]
            return pm.alpha_beta(self.asset_returns, self.weights, bench, rf_per_period=rf, ann_factor=ann)
        if nombre in ("ir", "information_ratio"):
            bench = kwargs["benchmark"]
            return pm.information_ratio(self.asset_returns, self.weights, bench, ann_factor=ann)

        raise ValueError(f"KPI no reconocido: {nombre}")


    def kpis_basicos(self, *, ann_factor: int | None = None, rf_per_period: float = 0.0) -> dict:
        """
        Atajo para obtener un dict con ER, Vol y Sharpe (por momentos).
        """
        return {
            "expected_return": self.kpi("exp_return", ann_factor=ann_factor),
            "volatility":      self.kpi("vol",         ann_factor=ann_factor),
            "sharpe_m":        self.kpi("sharpe_m",    ann_factor=ann_factor, rf_per_period=rf_per_period),
        }


    def pastel(self, pesos=None, min_weight: float = 1e-3) -> None:
        """
        Gráfica de pastel para los pesos del portafolio.

        Usa self.asset_returns.columns como universo y self.weights como pesos,
        delegando la lógica a portafolios.plots.pie.pastel_portfolio.
        """
        return pie.pastel_portfolio(self, pesos=pesos, min_weight=min_weight)
    
    def barras(self, pesos=None, min_weight: float = 0.0) -> None:
        """
        Gráfica de barras para los pesos del portafolio.

        Usa self.asset_returns.columns como universo y self.weights como pesos
        por defecto. Delegamos la lógica a portafolios.plots.barpy.barras_portfolio.
        """
        return bar.barras_portfolio(self, pesos=pesos, min_weight=min_weight)

    def bubbleplot(self, pesos=None, min_weight: float = 0.0):
        """
        Bubble plot ER vs VOL por activo.
        """
        return bubble.bubbleplot_portfolio(self, pesos=pesos, min_weight=min_weight)

    def corr_heatmap(self, kind: str = "correlation", round_decimals: int = 2) -> None:
        """
        Heatmap de correlación o covarianza usando las matrices del portafolio.
        kind: "correlation" o "covariance".
        """
        return corr_heatmap.corr_heatmap_portfolio(self, kind=kind, round_decimals=round_decimals)
