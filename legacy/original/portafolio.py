#el papá 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


class Portafolio:
    def __init__(self, data: pd.DataFrame, nombre: str = "Portafolio"):  
            
        """
        Inicializa un portafolio con datos de precios en formato MultiIndex por columnas:
        nivel 0 = ticker (activo), nivel 1 = campo (por ej. 'Close', 'Open', ...).

        Parámetros
        ----------
        data : pd.DataFrame
            DataFrame con columnas MultiIndex (ticker, campo) y un índice de fechas.
            Debe contener al menos el campo 'Close' para cada ticker.
        nombre : str
            Nombre legible del portafolio (para títulos de gráficas y exportes).

        Efectos
        -------
        - Crea una copia defensiva de `data`.
        - Valida estructura mínima (MultiIndex de 2 niveles y presencia de 'Close').
        - Inicializa atributos públicos y privados usados por otros métodos.
        """
        # copia y validación de la estructura 
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` debe ser un pandas.DataFrame.")
        data = data.copy()

        if not isinstance(data.columns, pd.MultiIndex) or data.columns.nlevels != 2:
         raise ValueError(
            "Las columnas deben ser MultiIndex de 2 niveles: (ticker, campo)."
        )

        lvl0, lvl1 = data.columns.levels[0], data.columns.levels[1]
        if "Close" not in data.columns.get_level_values(1):
            raise ValueError(
            "Falta el campo 'Close' en el segundo nivel de columnas."
        )
        
        self.data = data
        self.nombre = nombre
        # elementos 

        # --- splits de datos ---
        self.data_construct = None
        self.data_bt_train  = None
        self.data_bt_test   = None

        # métricas por tramo (opcional, útil para logging/comparación)
        self.expected_returns_construct = None
        self.volatility_construct       = None
        self.expected_returns_bt_train  = None
        self.volatility_bt_train        = None
        self.expected_returns_bt_test   = None
        self.volatility_bt_test         = None
        # universo base para construcción (para alinear pesos)
        self._construct_tickers: list[str] | None = None
        #universales 
        self.weights = None
        self.portfolioreturns = None
        
    def dividir(self,
                construct_start: str, construct_end: str,
                bt_train_start: str, bt_train_end: str,
                bt_test_start: str, bt_test_end: str):
        """
        Define tres tramos:
        - data_construct: para construir el portafolio final (entrenamiento definitivo)
        - data_bt_train: para calibración/validación interna (backtest-train)
        - data_bt_test: para prueba final fuera de muestra (backtest-test)
        """
        # 1) cortes por fecha
        self.data_construct = self.data.loc[construct_start:construct_end]
        self.data_bt_train = self.data.loc[bt_train_start:bt_train_end]
        self.data_bt_test  = self.data.loc[bt_test_start:bt_test_end] 
        
        # 2) universo de construcción (tickers) y orden canónico
        if self.data_construct is None or self.data_construct.empty:
            self._construct_tickers = []
        else:
            self._construct_tickers = self.data_construct.columns.get_level_values(0).unique().tolist()

         # 3) métricas por tramo (seguras ante ventanas vacías)
        def _safe_metrics(df):
            if df is None or df.empty:
                return None, None
            close = df.xs('Close', axis=1, level=1)
            rets = close.pct_change().dropna()
            if rets.empty:
                return None, None
            return rets.mean(), rets.std()

        self.expected_returns_construct, self.volatility_construct = _safe_metrics(self.data_construct)
        self.expected_returns_bt_train, self.volatility_bt_train   = _safe_metrics(self.data_bt_train)
        self.expected_returns_bt_test,  self.volatility_bt_test    = _safe_metrics(self.data_bt_test)

        return self.data_construct, self.data_bt_train, self.data_bt_test

    def calculate_expected_returns(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcula los **retornos esperados diarios** (promedios) por activo
        usando precios de cierre del `DataFrame` dado.

        Parámetros
        ----------
        data : pd.DataFrame
            DataFrame con columnas MultiIndex (ticker, campo) y un índice de fechas.
            Debe contener el campo 'Close' en el nivel 1.

        Returns
        -------
        pd.Series
            Serie con la media de los retornos diarios por activo (índice = tickers).
            Si `data` está vacío o no hay retornos (una sola fila), devuelve una Serie vacía.

        Notas
        -----
        - Usa **retornos aritméticos** (`pct_change()`), no logarítmicos.
        - La anualización se hace fuera (p. ej., con un helper de KPIs).
        """
        if data is None or data.empty:
            return pd.Series(dtype=float)
        if "Close" not in data.columns.get_level_values(1):
            raise ValueError("El DataFrame no contiene el campo 'Close' en sus columnas.")

        closeprices = data.xs('Close', axis=1, level=1)
        returns = closeprices.pct_change().dropna()
        if returns.empty:
            return pd.Series(dtype=float)
        return returns.mean()


    def calculate_volatility(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcula la **volatilidad diaria** (desviación estándar) por activo
        usando precios de cierre del `DataFrame` dado.

        Parámetros
        ----------
        data : pd.DataFrame
            DataFrame con columnas MultiIndex (ticker, campo) y un índice de fechas.
            Debe contener el campo 'Close' en el nivel 1.

        Returns
        -------
        pd.Series
            Serie con la desviación estándar de los retornos diarios por activo (índice = tickers).
            Si `data` está vacío o no hay retornos (una sola fila), devuelve una Serie vacía.

        Notas
        -----
        - Usa **retornos aritméticos** (`pct_change()`), no logarítmicos.
        - La anualización se hace fuera (multiplicando por `sqrt(252)`).
        """
        if data is None or data.empty:
            return pd.Series(dtype=float)
        if "Close" not in data.columns.get_level_values(1):
            raise ValueError("El DataFrame no contiene el campo 'Close' en sus columnas.")

        closeprices = data.xs('Close', axis=1, level=1)
        returns = closeprices.pct_change().dropna()
        if returns.empty:
            return pd.Series(dtype=float)
        return returns.std()


    def construir(self, pesos):
        """
        Fija los pesos del portafolio y calcula la serie de retornos **solo** sobre el
        set de construcción (`self.data_construct`), respetando el orden canónico
        (`self._construct_tickers`).

        Parámetros
        ----------
        pesos : array-like o pd.Series
            - Si es array-like, se asume el orden de `self._construct_tickers`.
            - Si es pd.Series, debe estar indexada por los tickers de construcción.

        Efectos
        -------
        - Valida que exista `data_construct`.
        - Alinea y normaliza pesos si su suma != 1 (sin cambiar su estructura relativa).
        - Guarda `self.weights` y compone `self.portfolioreturns` (retornos diarios del portafolio
        en la ventana de construcción).
        """
        # 1) checks básicos
        if self.data_construct is None or self.data_construct.empty:
            raise RuntimeError("Primero define los splits con dividir(...): data_construct está vacío.")

        close_construct = self.data_construct.xs('Close', axis=1, level=1)
        tickers = self._construct_tickers or close_construct.columns.tolist()

        # 2) aceptar array o Series indexada por ticker
        if isinstance(pesos, pd.Series):
            w = pesos.reindex(tickers)
            if w.isna().any():
                faltan = [t for t, v in w.items() if pd.isna(v)]
                raise ValueError(f"Faltan pesos para tickers de construcción: {faltan}")
            w = w.values.astype(float)
        else:
            w = np.asarray(pesos, dtype=float)
            if len(w) != len(tickers):
                raise ValueError(
                    f"Length of weights ({len(w)}) must match number of construction assets ({len(tickers)})."
                )

        # 3) validaciones de numericidad y normalización suave
        if not np.isfinite(w).all():
            raise ValueError("Los pesos contienen NaN o Inf.")
        # tolerancia para negativos microscópicos por redondeo
        w[w < 0] = np.maximum(w[w < 0], -1e-12)
        s = w.sum()
        if s == 0:
            raise ValueError("La suma de los pesos es 0; no se puede normalizar.")
        if not np.isclose(s, 1.0, atol=1e-8):
            w = w / s

        # 4) fijar pesos y calcular la serie de retornos sobre CONSTRUCT
        self.weights = w
        rets_construct = close_construct.loc[:, tickers].pct_change().dropna()
        self.portfolioreturns = rets_construct @ self.weights

        return self.portfolioreturns


    def compute_portfolio_returns(self, which: str = "construct"):
        """
        Devuelve la SERIE de retornos del portafolio para el tramo indicado.
        which ∈ {"construct","bt_train","bt_test"}.
        """
        if which == "construct":
            data = self.data_construct
        elif which == "bt_train":
            data = self.data_bt_train
        elif which == "bt_test":
            data = self.data_bt_test
        else:
            raise ValueError("which debe ser 'construct', 'bt_train' o 'bt_test'.")

        if data is None or data.empty:
            self.portfolioreturns = None
            return None

        close = data.xs('Close', axis=1, level=1)
        tickers = self._construct_tickers or close.columns.tolist()
        rets = close.loc[:, tickers].pct_change().dropna()
        self.portfolioreturns = rets @ self.weights
        return self.portfolioreturns


    def mean_portfolio_return(self, which: str = "construct", annualize: bool = False, freq: int = 252):
        """
        Devuelve el PROMEDIO de los retornos (diario por defecto) del portafolio.
        Si annualize=True, devuelve el retorno anualizado.
        """
        rs = self.portfolioreturns
        # si la serie aún no está calculada o el tramo cambió, la calculamos
        if rs is None:
            rs = self.compute_portfolio_returns(which)
        if rs is None or rs.empty:
            return None

        mean_daily = rs.mean()
        if not annualize:
            return mean_daily
        return (1 + mean_daily)**freq - 1

    def portfolio_volatility(self, which: str = "construct", annualize: bool = False, freq: int = 252):
        """
        Devuelve la volatilidad (desviación estándar) de la SERIE de retornos del portafolio
        para el tramo indicado.

        Parámetros
        ----------
        which : {"construct","bt_train","bt_test"}
            Tramo sobre el que calcular la volatilidad.
        annualize : bool
            Si True, devuelve volatilidad anualizada (σ_diaria * sqrt(freq)).
        freq : int
            Frecuencia de trading para anualizar (por defecto 252).

        Returns
        -------
        float | None
            Volatilidad (diaria o anualizada). Devuelve None si la serie está vacía.
        """
        # Asegurar que la serie de retornos del tramo esté disponible
        rs = self.portfolioreturns
        if rs is None:
            rs = self.compute_portfolio_returns(which)
        else:
            # Si la serie existente no corresponde al tramo pedido, recalcular
            # (opcional: puedes guardar un flag del tramo actual si quieres ser estricto)
            if rs is None or rs.empty:
                rs = self.compute_portfolio_returns(which)

        if rs is None or rs.empty:
            return None

        vol = rs.std()
        if annualize:
            vol *= np.sqrt(freq)
        return float(vol)

    def bt_train(self, pesos):
        """
        Evalúa los mismos pesos del portafolio en la ventana BT-TRAIN.
        Devuelve la SERIE de retornos del portafolio en ese tramo.
        """
        if self.data_bt_train is None or self.data_bt_train.empty:
            print("Advertencia: la ventana de BT‑train está vacía; se omite evaluación.")
            return None

        close_bt = self.data_bt_train.xs('Close', axis=1, level=1)

        # Alinear al universo de construcción
        construct_tickers = self._construct_tickers or []
        cols = [c for c in close_bt.columns if c in construct_tickers]
        if not cols:
            print("BT‑train no comparte activos con construcción.")
            return None

        # Reindexar/renormalizar pesos si faltan activos
        idx = [construct_tickers.index(c) for c in cols]
        w = np.asarray(pesos, dtype=float)[idx]
        s = w.sum()
        if s == 0:
            print("Pesos nulos tras filtrar columnas en BT‑train.")
            return None
        w = w / s

        rs = close_bt.loc[:, cols].pct_change().dropna() @ w
        return rs


    def bt_test(self, pesos):
        """
        Evalúa los mismos pesos del portafolio en la ventana BT-TEST.
        Devuelve la SERIE de retornos del portafolio en ese tramo.
        """
        if self.data_bt_test is None or self.data_bt_test.empty:
            print("Advertencia: la ventana de BT‑test está vacía; se omite evaluación.")
            return None

        close_bt = self.data_bt_test.xs('Close', axis=1, level=1)

        construct_tickers = self._construct_tickers or []
        cols = [c for c in close_bt.columns if c in construct_tickers]
        if not cols:
            print("BT‑test no comparte activos con construcción.")
            return None

        idx = [construct_tickers.index(c) for c in cols]
        w = np.asarray(pesos, dtype=float)[idx]
        s = w.sum()
        if s == 0:
            print("Pesos nulos tras filtrar columnas en BT‑test.")
            return None
        w = w / s

        rs = close_bt.loc[:, cols].pct_change().dropna() @ w
        return rs

    def mostrar_pesos(self, pesos=None, top: int | None = None, as_dataframe: bool = False, decimals: int = 4):
        """
        Imprime (y opcionalmente devuelve) los pesos del portafolio alineados al
        universo de CONSTRUCCIÓN (`self._construct_tickers`).

        Parámetros
        ----------
        pesos : array-like | pd.Series | None
            - None: usa `self.weights` ya fijados por `construir(...)`.
            - array-like: se asume el orden de `self._construct_tickers`.
            - pd.Series: debe estar indexada por ticker; se reindexa a construcción.
        top : int | None
            Si se indica, muestra solo los 'top' mayores pesos (orden descendente).
        as_dataframe : bool
            Si True, devuelve un DataFrame con columnas ['ticker','weight','pct'].
        decimals : int
            Decimales para impresión del peso en formato flotante.

        Returns
        -------
        None | pd.DataFrame
            Devuelve DataFrame si `as_dataframe=True`.
        """
        # --- definir universo de construcción ---
        construct_tickers = self._construct_tickers or []
        if not construct_tickers:
            print("No hay universo de construcción definido. Llama primero a dividir(...).")
            return None

        # --- obtener/alinéar pesos ---
        if pesos is None:
            if getattr(self, "weights", None) is None:
                print("No hay pesos en el objeto. Pasa 'pesos' o llama a construir(...).")
                return None
            w = np.asarray(self.weights, dtype=float)
            if len(w) != len(construct_tickers):
                print("Los pesos guardados no coinciden con el universo de construcción.")
                return None
            tickers = construct_tickers
        elif isinstance(pesos, pd.Series):
            s = pesos.reindex(construct_tickers)
            if s.isna().any():
                faltan = [t for t, v in s.items() if pd.isna(v)]
                raise ValueError(f"Faltan pesos para tickers de construcción: {faltan}")
            w = s.values.astype(float)
            tickers = construct_tickers
        else:
            w = np.asarray(pesos, dtype=float)
            if len(w) != len(construct_tickers):
                raise ValueError(
                    f"Length of weights ({len(w)}) debe coincidir con activos de construcción ({len(construct_tickers)})."
                )
            tickers = construct_tickers

        # --- limpieza numérica y normalización suave ---
        if not np.isfinite(w).all():
            raise ValueError("Los pesos contienen NaN o Inf.")
        w[w < 0] = np.maximum(w[w < 0], -1e-12)  # clamp negativos mínimos
        ssum = w.sum()
        if ssum == 0:
            raise ValueError("Suma de pesos = 0; no se puede normalizar.")
        if not np.isclose(ssum, 1.0, atol=1e-8):
            w = w / ssum

        # --- armar tabla ordenada por peso desc. ---
        df = pd.DataFrame({"ticker": tickers, "weight": w})
        df["pct"] = (df["weight"] * 100).round(2)
        df = df.sort_values("weight", ascending=False, kind="mergesort").reset_index(drop=True)

        if isinstance(top, int) and top > 0:
            df_print = df.head(top)
        else:
            df_print = df

        # --- salida en texto ---
        for _, row in df_print.iterrows():
            print(f"acción: {row['ticker']}, peso asignado: {row['weight']:.{decimals}f}  ({row['pct']:.2f}%)")

        if top and top < len(df):
            resto = len(df) - top
            pct_resto = df["pct"].iloc[top:].sum()
            print(f"... y {resto} activos más sumando {pct_resto:.2f}%")

        return df if as_dataframe else None

    def pastel(self, pesos=None, min_weight: float = 1e-3):
        """
        Grafica un pastel (donut) de los pesos de CONSTRUCCIÓN.
        - Acepta pesos explícitos (array o Series) o usa self.weights si ya construiste.
        - Alinea a self._construct_tickers y rellena faltantes con 0 (p. ej., Markowitz).
        - Filtra pesos muy pequeños para legibilidad (min_weight).
        - Guarda el HTML en ./plots/ y también lo muestra en el navegador.
        """
        # --- universo de construcción ---
        construct_tickers = self._construct_tickers or []
        if not construct_tickers:
            print("No hay universo de construcción. Llama primero a dividir(...).")
            return

        # --- obtener / alinear pesos ---
        if pesos is None:
            if getattr(self, "weights", None) is None:
                print("No hay pesos en el objeto. Pasa 'pesos' o llama a construir(...).")
                return
            w = np.asarray(self.weights, dtype=float)
            if len(w) != len(construct_tickers):
                print("Los pesos guardados no coinciden con el universo de construcción.")
                return
            tickers = construct_tickers
        elif isinstance(pesos, pd.Series):
            # reindex a universo de construcción; faltantes = 0 (caso Markowitz)
            s = pesos.reindex(construct_tickers).fillna(0.0)
            w = s.values.astype(float)
            tickers = construct_tickers
        else:
            w = np.asarray(pesos, dtype=float)
            if len(w) != len(construct_tickers):
                raise ValueError(
                    f"Length of weights ({len(w)}) debe coincidir con activos de construcción ({len(construct_tickers)})."
                )
            tickers = construct_tickers

        # limpieza y normalización suave
        if not np.isfinite(w).all():
            raise ValueError("Los pesos contienen NaN o Inf.")
        w[w < 0] = np.maximum(w[w < 0], -1e-12)
        ssum = w.sum()
        if ssum == 0:
            print("Suma de pesos = 0; nada que graficar.")
            return
        if not np.isclose(ssum, 1.0, atol=1e-8):
            w = w / ssum

        # filtrar pesos pequeños para legibilidad
        mask = w > min_weight
        pesos_filtrados = w[mask].tolist()
        tickers_filtrados = [t for t, m in zip(tickers, mask) if m]

        # tamaños y texto según número de activos
        n_activos = len(tickers)
        if n_activos <= 20:
            width, height = 1200, 800
            text_size = 12; legend_size = 10
        elif n_activos <= 40:
            width, height = 1400, 900
            text_size = 11; legend_size = 9
        else:
            width, height = 1600, 1000
            text_size = 10; legend_size = 8

        # detectar equiponderado (naive) aproximado
        es_naive = False
        if len(pesos_filtrados) > 0:
            p = np.array(pesos_filtrados, dtype=float)
            # coeficiente de variación bajo → casi iguales
            if p.mean() > 0 and (p.std() / p.mean()) < 0.02 and len(p) > 10:
                es_naive = True

        # ajustar agujero
        if len(pesos_filtrados) < len(tickers) * 0.5:
            hole_size = 0.1
        elif es_naive:
            hole_size = 0.2
        else:
            hole_size = 0.3

        # colores / pull
        if es_naive:
            base_colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
            reps = (len(tickers_filtrados) // len(base_colors)) + 1
            colors = (base_colors * reps)[:len(tickers_filtrados)]
            pull_effect = [0.05] * len(tickers_filtrados)
        else:
            colors = None
            pull_effect = [0.1 if p > 0.05 else 0 for p in pesos_filtrados]

        fig = go.Figure(data=[go.Pie(
            labels=tickers_filtrados,
            values=pesos_filtrados,
            hole=hole_size,
            textinfo='label+percent',
            textposition='outside',
            textfont=dict(size=text_size),
            marker=dict(line=dict(color='white', width=1), colors=colors),
            rotation=45,
            pull=pull_effect
        )])

        # título
        activos_excluidos = len(tickers) - len(tickers_filtrados)
        if activos_excluidos > 0:
            title_text = (f'Pesos del {self.nombre}'
                        f'<br><sub>Se muestran {len(tickers_filtrados)} de {len(tickers)} activos '
                        f'(excluidos {activos_excluidos} con peso ≤ {min_weight:.2%})</sub>')
        elif es_naive and pesos_filtrados:
            title_text = (f'Pesos del {self.nombre}'
                        f'<br><sub>Portfolio equiponderado: {len(tickers_filtrados)} activos '
                        f'≈ {pesos_filtrados[0]:.1%} cada uno</sub>')
        else:
            title_text = f'Pesos del {self.nombre}'

        fig.update_layout(
            title=title_text, showlegend=True, width=width, height=height,
            title_font_size=20, font=dict(size=legend_size),
            margin=dict(l=50, r=50, t=100, b=50),
            plot_bgcolor='white', paper_bgcolor='white',
            legend=dict(orientation="v", yanchor="top", y=0.99,
                        xanchor="left", x=1.02, font=dict(size=legend_size))
        )

        # guardar en ./plots y mostrar
        safe_name = self.nombre.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("\\", "_")
        plots_dir = Path.cwd() / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path = plots_dir / f'portfolio_pastel_{safe_name}.html'
        fig.write_html(str(out_path))
        fig.show()

    def barras(self, pesos=None, min_weight: float = 0.0):
        """
        Grafica barras de los pesos (universo de CONSTRUCCIÓN).
        - Si `pesos` es None, usa `self.weights`.
        - Alinea a `self._construct_tickers`; si faltan pesos en una Series, rellena con 0.
        - Normaliza si la suma != 1. Puede filtrar pesos muy pequeños (min_weight).
        - Guarda HTML en ./plots/ y muestra en navegador.
        """
        construct_tickers = self._construct_tickers or []
        if not construct_tickers:
            print("No hay universo de construcción. Llama primero a dividir(...).")
            return

        # --- obtener / alinear pesos ---
        if pesos is None:
            if getattr(self, "weights", None) is None:
                print("No hay pesos en el objeto. Pasa 'pesos' o llama a construir(...).")
                return
            w = np.asarray(self.weights, dtype=float)
            if len(w) != len(construct_tickers):
                print("Los pesos guardados no coinciden con el universo de construcción.")
                return
            tickers = construct_tickers
        elif isinstance(pesos, pd.Series):
            s = pesos.reindex(construct_tickers).fillna(0.0)
            w = s.values.astype(float)
            tickers = construct_tickers
        else:
            w = np.asarray(pesos, dtype=float)
            if len(w) != len(construct_tickers):
                raise ValueError(
                    f"Length of weights ({len(w)}) debe coincidir con activos de construcción ({len(construct_tickers)})."
                )
            tickers = construct_tickers

        # --- limpieza / normalización ---
        if not np.isfinite(w).all():
            raise ValueError("Los pesos contienen NaN o Inf.")
        w[w < 0] = np.maximum(w[w < 0], -1e-12)
        ssum = w.sum()
        if ssum == 0:
            print("Suma de pesos = 0; nada que graficar.")
            return
        if not np.isclose(ssum, 1.0, atol=1e-8):
            w = w / ssum

        # --- filtrar pesos muy pequeños para legibilidad (opcional) ---
        mask = w > min_weight
        w_plot = w[mask]
        t_plot = [t for t, m in zip(tickers, mask) if m]

        # --- gráfico ---
        fig = go.Figure(data=[go.Bar(
            x=t_plot,
            y=w_plot,
            text=[f'{p:.1%}' for p in w_plot],
            textposition='auto'
        )])

        fig.update_layout(
            title=f'Distribución de Pesos del {self.nombre}',
            xaxis_title='Activos',
            yaxis_title='Peso',
            width=1400,
            height=700,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # --- guardar y mostrar ---
        safe_name = self.nombre.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("\\", "_")
        plots_dir = Path.cwd() / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path = plots_dir / f'portfolio_barras_{safe_name}.html'
        fig.write_html(str(out_path))
        fig.show()

    def bubbleplot(self, pesos=None, min_weight: float = 0.0):
        """
        Bubble plot (Plotly) de Rendimiento esperado vs Volatilidad por ACTIVO,
        usando el universo de CONSTRUCCIÓN. El tamaño de cada burbuja es el peso.

        Parámetros
        ----------
        pesos : array-like | pd.Series | None
            - None: usa self.weights (después de construir).
            - array-like: se asume el orden de self._construct_tickers.
            - pd.Series: se reindexa a self._construct_tickers; faltantes -> 0 (p. ej. Markowitz).
        min_weight : float
            Umbral para filtrar pesos muy pequeños (no se muestran).

        Efectos
        -------
        - Calcula ER y VOL por activo en la ventana de construcción.
        - Alinea pesos y métricas al universo de construcción.
        - Normaliza pesos si la suma != 1.
        - Guarda el HTML en ./plots/bubble_plot_{nombre}.html y abre la figura.
        """
        # --- universo de construcción ---
        construct_tickers = self._construct_tickers or []
        if not construct_tickers:
            print("No hay universo de construcción. Llama primero a dividir(...).")
            return

        if self.data_construct is None or self.data_construct.empty:
            print("data_construct está vacío. Revisa tus fechas.")
            return

        # --- métricas por activo en CONSTRUCT ---
        close = self.data_construct.xs('Close', axis=1, level=1)
        # asegurar columnas en el mismo orden del universo
        cols = [c for c in construct_tickers if c in close.columns]
        if not cols:
            print("No hay intersección de tickers entre construct y datos de cierre.")
            return

        rets = close.loc[:, cols].pct_change().dropna()
        if rets.empty:
            print("No hay retornos en la ventana de construcción (muy pocos datos).")
            return

        ereturns = rets.mean()      # pd.Series index=tickers
        vols     = rets.std()       # pd.Series index=tickers

        # --- obtener / alinear pesos ---
        if pesos is None:
            if getattr(self, "weights", None) is None:
                print("No hay pesos en el objeto. Pasa 'pesos' o llama a construir(...).")
                return
            w = np.asarray(self.weights, dtype=float)
            if len(w) != len(construct_tickers):
                print("Los pesos guardados no coinciden con el universo de construcción.")
                return
            w = pd.Series(w, index=construct_tickers)
        elif isinstance(pesos, pd.Series):
            # reindex a universo de construcción; faltantes = 0
            w = pesos.reindex(construct_tickers).fillna(0.0).astype(float)
        else:
            w_arr = np.asarray(pesos, dtype=float)
            if len(w_arr) != len(construct_tickers):
                raise ValueError(
                    f"Length of weights ({len(w_arr)}) debe coincidir con activos de construcción ({len(construct_tickers)})."
                )
            w = pd.Series(w_arr, index=construct_tickers)

        # limpieza / normalización
        if not np.isfinite(w.values).all():
            raise ValueError("Los pesos contienen NaN o Inf.")
        w[w < 0] = np.maximum(w[w < 0], -1e-12)
        ssum = float(w.sum())
        if ssum == 0:
            print("Suma de pesos = 0; nada que graficar.")
            return
        if not np.isclose(ssum, 1.0, atol=1e-8):
            w = w / ssum

        # --- alinear ER/VOL a universo y filtrar por min_weight ---
        ereturns = ereturns.reindex(construct_tickers)
        vols     = vols.reindex(construct_tickers)

        mask = w.values > float(min_weight)
        if not mask.any():
            print(f"Todos los pesos son ≤ {min_weight:.4f}; nada que graficar.")
            return
        w_plot   = w.values[mask]
        x_plot   = ereturns.values[mask]
        y_plot   = vols.values[mask]
        labels   = np.array(construct_tickers)[mask]

        # --- tamaños de burbuja (escala suave) ---
        # raíz para que no dominen los grandes; escala automática por cantidad
        n_pts = len(w_plot)
        base_scale = 1800 if n_pts <= 25 else (1400 if n_pts <= 60 else 1000)
        sizes = np.maximum(8.0, np.sqrt(w_plot) * base_scale)  # mínimo visible 8

        # --- figura plotly ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_plot, y=y_plot,
            mode='markers+text' if n_pts <= 25 else 'markers',
            marker=dict(
                size=sizes,
                opacity=0.7,
                line=dict(color='black', width=1)
            ),
            text=labels if n_pts <= 25 else None,
            textposition='middle center',
            hovertemplate="<b>%{text}</b><br>ER: %{x:.4f}<br>VOL: %{y:.4f}<br>Peso: %{customdata:.2%}<extra></extra>",
            customdata=w_plot,
            name='Activos'
        ))

        fig.update_layout(
            title=f'Bubble Plot: Returns vs Volatility - {self.nombre}',
            xaxis_title='Expected Returns (diario)',
            yaxis_title='Volatility (diaria)',
            width=1200, height=900,
            showlegend=False,
            plot_bgcolor='white', paper_bgcolor='white'
        )

        # --- guardar y mostrar ---
        safe_name = self.nombre.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("\\", "_")
        plots_dir = Path.cwd() / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path = plots_dir / f'bubble_plot_{safe_name}.html'
        fig.write_html(str(out_path))
        fig.show()


    def mc(self,pesos):
        closeprices = self.data.xs('Close', axis=1, level=1)
        logreturns = np.log(closeprices / closeprices.shift(1)).dropna()

        mean_returns = logreturns.mean()
        cov_matrix = logreturns.cov()
        n_simulations = 5000
        n_days = 252
        simulations = np.zeros((n_simulations, n_days))

        for i in range(n_simulations):
            daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, size=n_days)
            price_paths = np.exp(np.cumsum(daily_returns, axis=0))
            price_paths = price_paths * np.array([1] * len(self.tickers))
            portfolio_value = price_paths @ pesos
            simulations[i, :] = portfolio_value

        fig = go.Figure()
        for i in range(n_simulations):
            fig.add_trace(go.Scatter(
                x=np.arange(n_days),
                y=simulations[i, :],
                mode='lines',
                line=dict(color='blue'),
                opacity=0.1
            ))

        fig.update_layout(
            title='Simulación Monte Carlo del Portafolio',
            xaxis_title='Días',
            yaxis_title='Valor del Portafolio',
            showlegend=False
        )
        fig.show()

    def bt(self,pesos):
        bt_returns = self.data_bt.xs('Close', axis=1, level=1).pct_change().dropna() @ pesos
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bt_returns.index,
            y=(1 + bt_returns).cumprod(),
            mode='lines',
            name='Backtest Portfolio'
        ))
        fig.update_layout(title='Crecimiento del Portafolio en el Backtest')
        fig.show()
    
    def risk(self,pesos):
        pass

    
    


    
    def bubbleplot_matplotlib(self, weights):
        tickers = self.data.columns.get_level_values(0).unique()
        closeprices = self.data.xs('Close', axis=1, level=1)
        returns = closeprices.pct_change().dropna()

        ereturns = returns.mean()
        vols = returns.std()

        # Crear gráfico de burbujas con Plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=ereturns.values,
            y=vols.values,
            mode='markers+text',
            marker=dict(
                size=weights * 2000,
                color='cornflowerblue',
                opacity=0.7,
                line=dict(color='black', width=1)
            ),
            text=tickers,
            textposition='middle center',
            name='Activos'
        ))
        
        fig.update_layout(
            title=f'Bubble Plot: Returns vs Volatility - {self.nombre}',
            xaxis_title='Expected Returns',
            yaxis_title='Volatility',
            width=1200,
            height=900,
            title_font_size=20,
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            showlegend=False
        )
        
        # Guardar como HTML
        safe_name = self.nombre.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("\\", "_")
        plots_dir = Path(__file__).resolve().parents[1] / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path = plots_dir / f'bubble_plot_{safe_name}.html'
        fig.write_html(str(out_path))
        
        # Mostrar en navegador
        fig.show()

    def correlacion_y_dendograma(self):
        close = self.data.xs('Close', axis=1, level=1)
        corr_original = close.pct_change().dropna().corr()

        # Matriz de correlaciones original con Plotly
        # Solo mostrar valores de correlación para activos importantes o en hover
        # Crear matriz de texto solo para correlaciones extremas
        text_matrix = corr_original.copy()
        text_matrix = text_matrix.applymap(lambda x: f'{x:.2f}' if abs(x) > 0.7 else '')  # Solo mostrar correlaciones > 0.7 o < -0.7
        
        fig1 = go.Figure(data=go.Heatmap(
            z=corr_original.values,
            x=corr_original.columns,
            y=corr_original.index,
            colorscale='RdBu',
            zmid=0,
            text=text_matrix.values,  # Mostrar solo correlaciones importantes
            texttemplate="%{text}",
            textfont={"size": 9, "color": "black"},
            hovertemplate='<b>%{y} vs %{x}</b><br>' +
                         'Correlación: %{z:.3f}<br>' +
                         '<extra></extra>',
            hoverongaps=False
        ))
        
        fig1.update_layout(
            title="Matriz de Correlaciones (Original)<br><sub>Hover sobre las celdas para ver valores exactos</sub>",
            width=1000,
            height=1000,
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=8),
                showticklabels=True
            ),
            yaxis=dict(
                tickfont=dict(size=8),
                showticklabels=True
            )
        )
        
        # Guardar y mostrar
        plots_dir = Path(__file__).resolve().parents[1] / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path1 = plots_dir / 'correlacion_original.html'
        fig1.write_html(str(out_path1))
        fig1.show()

        # Reordenar usando cuasidiagonalización
        dist = 1 - corr_original
        linked = linkage(squareform(dist), method='single')
        dendro = dendrogram(linked, no_plot=True)
        order = dendro['leaves']
        corr_reordered = corr_original.iloc[order, :].T.iloc[order, :]

        # Matriz de correlaciones reorganizada
        # Crear matriz de texto solo para correlaciones extremas
        text_matrix_reordered = corr_reordered.copy()
        text_matrix_reordered = text_matrix_reordered.applymap(lambda x: f'{x:.2f}' if abs(x) > 0.7 else '')  # Solo mostrar correlaciones > 0.7 o < -0.7
        
        fig2 = go.Figure(data=go.Heatmap(
            z=corr_reordered.values,
            x=corr_reordered.columns,
            y=corr_reordered.index,
            colorscale='RdBu',
            zmid=0,
            text=text_matrix_reordered.values,  # Mostrar solo correlaciones importantes
            texttemplate="%{text}",
            textfont={"size": 9, "color": "black"},
            hovertemplate='<b>%{y} vs %{x}</b><br>' +
                         'Correlación: %{z:.3f}<br>' +
                         '<extra></extra>',
            hoverongaps=False
        ))
        
        fig2.update_layout(
            title="Matriz de Correlaciones (Reordenada)<br><sub>Después de la cuasidiagonalización - Hover sobre las celdas para ver valores exactos</sub>",
            width=1000,
            height=1000,
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=8),
                showticklabels=True
            ),
            yaxis=dict(
                tickfont=dict(size=8),
                showticklabels=True
            )
        )
        
        # Guardar y mostrar
        plots_dir = Path(__file__).resolve().parents[1] / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path2 = plots_dir / 'correlacion_reordenada.html'
        fig2.write_html(str(out_path2))
        fig2.show()

        # Dendograma con Plotly usando scipy.dendrogram
        # Crear el dendograma real usando scipy
        from scipy.cluster.hierarchy import dendrogram
        
        # Crear figura para el dendograma
        fig3 = go.Figure()
        
        # Obtener las coordenadas del dendograma
        dendro_data = dendrogram(linked, no_plot=True, labels=close.columns.tolist())
        
        # Extraer las coordenadas de las líneas del dendograma
        icoord = dendro_data['icoord']  # Coordenadas x de las líneas
        dcoord = dendro_data['dcoord']  # Coordenadas y de las líneas
        
        # Agregar cada línea del dendograma
        for i in range(len(icoord)):
            # Convertir coordenadas de scipy a coordenadas de plotly
            x_coords = icoord[i]
            y_coords = dcoord[i]
            
            # Agregar línea horizontal
            fig3.add_trace(go.Scatter(
                x=x_coords,
                y=[y_coords[1], y_coords[1]],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Agregar líneas verticales
            fig3.add_trace(go.Scatter(
                x=[x_coords[0], x_coords[0]],
                y=[y_coords[0], y_coords[1]],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig3.add_trace(go.Scatter(
                x=[x_coords[2], x_coords[2]],
                y=[y_coords[2], y_coords[1]],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Agregar etiquetas de los activos en el eje X
        leaf_positions = dendro_data['leaves']
        leaf_labels = [close.columns[i] for i in leaf_positions]
        
        fig3.add_trace(go.Scatter(
            x=list(range(len(leaf_labels))),
            y=[0] * len(leaf_labels),
            mode='text',
            text=leaf_labels,
            textposition='bottom center',
            textfont=dict(size=8, color='black'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig3.update_layout(
            title="Dendograma de Clustering Jerárquico<br><sub>Muestra la estructura de agrupación de activos por correlación</sub>",
            xaxis_title="Activos (ordenados por clustering)",
            yaxis_title="Distancia de Correlación",
            width=1400,
            height=800,
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-1, len(leaf_labels)]
            ),
            yaxis=dict(
                showgrid=True,
                zeroline=True,
                zerolinecolor='lightgray',
                gridcolor='lightgray'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Guardar y mostrar
        plots_dir = Path(__file__).resolve().parents[1] / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path3 = plots_dir / 'dendograma.html'
        fig3.write_html(str(out_path3))
        fig3.show()

    @property
    def retornos(self):
        closeprices = self.data.xs('Close', axis=1, level=1)
        return closeprices.pct_change().dropna()

    @property
    def tickers(self):
        return self.data.columns.get_level_values(0).unique().to_list() 