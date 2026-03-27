# main_smoke_naive.py
import numpy as np
import pandas as pd
from pathlib import Path

from original.portafolio import Portafolio
from original.naive import NaivePortafolio  # hereda Portafolio con tus cambios

def main():
    # 1) cargar datos
    data_path = Path(__file__).with_name('data_clean_stock_data.csv')
    df = pd.read_csv(str(data_path), header=[0,1], index_col=0, parse_dates=True)

    # 2) definir ventanas (ejemplo)
    construct_start, construct_end = '2021-01-01', '2022-12-31'
    bt_train_start, bt_train_end   = '2019-01-01', '2020-12-31'
    bt_test_start, bt_test_end     = '2023-01-01', '2023-12-31'

    # 3) instanciar
    pf = NaivePortafolio(df)

    # 4) triple split
    pf.dividir(
        construct_start, construct_end,
        bt_train_start, bt_train_end,
        bt_test_start, bt_test_end
    )

    # 5) pesos naive en el universo de construcción
    n = len(pf._construct_tickers)
    assert n > 0, "Sin tickers en el tramo de construcción."
    pesos = np.ones(n) / n

    # 6) construir SOLO sobre CONSTRUCT (devuelve serie de retornos construct)
    rs_construct = Portafolio.construir(pf, pesos)

    # 7) evaluar con los mismos pesos en BT-train y BT-test
    rs_bt_train = pf.bt_train(pesos)
    rs_bt_test  = pf.bt_test(pesos)

    # 8) checks básicos (sin KPIs)
    print(f"[OK] Tickers en construct: {n}")
    print(f"[OK] Suma de pesos: {pesos.sum():.6f}")
    print(f"[OK] Retornos construct: {len(rs_construct) if rs_construct is not None else 0} observaciones")
    print(f"[OK] Retornos bt_train : {len(rs_bt_train) if rs_bt_train is not None else 0} observaciones")
    print(f"[OK] Retornos bt_test  : {len(rs_bt_test)  if rs_bt_test  is not None else 0} observaciones")

    # === Verificaciones adicionales de los métodos trabajados hoy ===
    print("\n[CHECK] Métricas por ACTIVO (construct/bt_train/bt_test):")
    er_c = pf.expected_returns_construct
    vol_c = pf.volatility_construct
    er_tr = pf.expected_returns_bt_train
    vol_tr = pf.volatility_bt_train
    er_te = pf.expected_returns_bt_test
    vol_te = pf.volatility_bt_test
    print("  construct  -> ER:", None if er_c is None else f"{er_c.shape}", " VOL:", None if vol_c is None else f"{vol_c.shape}")
    print("  bt_train   -> ER:", None if er_tr is None else f"{er_tr.shape}", " VOL:", None if vol_tr is None else f"{vol_tr.shape}")
    print("  bt_test    -> ER:", None if er_te is None else f"{er_te.shape}", " VOL:", None if vol_te is None else f"{vol_te.shape}")

    # Serie del portafolio (construct ya la calculó construir)
    print("\n[CHECK] compute_portfolio_returns (bt_train/bt_test):")
    rs_bt_train2 = pf.compute_portfolio_returns("bt_train")
    rs_bt_test2  = pf.compute_portfolio_returns("bt_test")
    print("  bt_train serie len:", 0 if rs_bt_train2 is None else len(rs_bt_train2))
    print("  bt_test  serie len:", 0 if rs_bt_test2  is None else len(rs_bt_test2))

    # Promedio y volatilidad (diaria y anualizada) usando la misma serie
    print("\n[CHECK] mean_portfolio_return / portfolio_volatility (construct):")
    mean_c = pf.mean_portfolio_return("construct")
    vol_c_p = pf.portfolio_volatility("construct")
    mean_c_ann = pf.mean_portfolio_return("construct", annualize=True)
    vol_c_ann  = pf.portfolio_volatility("construct", annualize=True)
    print(f"  mean_d={None if mean_c is None else round(float(mean_c),6)}  vol_d={None if vol_c_p is None else round(float(vol_c_p),6)}")
    print(f"  mean_ann={None if mean_c_ann is None else round(float(mean_c_ann),6)}  vol_ann={None if vol_c_ann is None else round(float(vol_c_ann),6)}")

    print("\n[CHECK] mean/vol en BT-train y BT-test:")
    for which in ["bt_train","bt_test"]:
        m = pf.mean_portfolio_return(which)
        v = pf.portfolio_volatility(which)
        print(f"  {which:8s} -> mean_d={None if m is None else round(float(m),6)}  vol_d={None if v is None else round(float(v),6)}")

    print("\n[CHECK] Pesos top 10:")
    pf.mostrar_pesos(top=10)

    print("\n[CHECK] Pastel (top visual):")
    pf.pastel()     

    print("\n[CHECK] Barras:")
    pf.barras() 

    print("\n[CHECK] Bubble plot:")
    pf.bubbleplot()  # usa self.weights y data_construct


if __name__ == "__main__":
    main()
