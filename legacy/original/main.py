#librerias 
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Configurar codificación para Windows
if sys.platform.startswith('win'):
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

#otras clases ¿tengo que poner todas asi?
from original.naive import NaivePortafolio
from original.hrp_style import HRPStyle
from original.markowitz import MarkowitzPortafolio
from distance import distdecorr
from original.clustering import SingleLinkage
from allocation import Naiverp

def main():
    try:
        print(" Iniciando librería de portafolios...")
        
        # leer los datos usando ruta relativa al archivo actual
        data_path = Path(__file__).with_name('data_clean_stock_data.csv')
        print(f"Leyendo datos desde: {data_path}")
        
        if not data_path.exists():
            print(f" Error: No se encontró el archivo {data_path}")
            return
            
        df = pd.read_csv(str(data_path), header=[0,1], index_col=0, parse_dates=True)
        print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        start_train_date = '2021-01-01'  
        end_train_date = '2023-01-01'  
        start_bt_date = '2020-01-01'    
        end_bt_date = '2020-12-31' 
        
        print(f"Período entrenamiento: {start_train_date} a {end_train_date}")
        print(f" Período backtesting: {start_bt_date} a {end_bt_date}")
        
        print("\n" + "="*50)
        print("=== PORTFOLIO NAIVE (1/N) ===")
        print("="*50)
        
        try:
            portafolionaive = NaivePortafolio(df)
            pesos_naive = portafolionaive.construir(start_train_date, end_train_date, start_bt_date, end_bt_date) 
            print(f" Pesos naive calculados: {len(pesos_naive)} activos")
            
            # Mostrar pesos primero
            portafolionaive.mostrar_pesos(pesos_naive)
            
            # Luego generar gráficos (se abrirán en pestañas del navegador)
            print("Generando gráfico de pastel...")
            portafolionaive.pastel(pesos_naive)
            
            print("Generando gráfico de barras...")
            portafolionaive.barras(pesos_naive)
            
            print("Portfolio Naive completado exitosamente")
            
        except Exception as e:
            print(f" Error en Portfolio Naive: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*50)
        print("=== PORTFOLIO MARKOWITZ ===")
        print("="*50)
        
        try:
            portafolio_markowitz = MarkowitzPortafolio(df)
            print("Optimizando portfolio Markowitz...")
            pesos_markowitz = portafolio_markowitz.construir(start_train_date, end_train_date, start_bt_date, end_bt_date)
            
            if pesos_markowitz is not None:
                print(f"Pesos Markowitz calculados: {len(pesos_markowitz)} activos")
                portafolio_markowitz.mostrar_pesos(pesos_markowitz)
                
                print("Generando gráfico de pastel...")
                portafolio_markowitz.pastel(pesos_markowitz)
                
                print("Generando gráfico de barras...")
                portafolio_markowitz.barras(pesos_markowitz)
                
                print("Generando gráfico de burbujas...")
                portafolio_markowitz.bubbleplot_matplotlib(pesos_markowitz)
                
                print(" Portfolio Markowitz completado exitosamente")
            else:
                print(" Portfolio Markowitz no pudo calcular pesos")
                
        except Exception as e:
            print(f"Error en Portfolio Markowitz: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*50)
        print("=== PORTFOLIO HRP ===")
        print("="*50)
        
        try:
            print("🔧 Configurando componentes HRP...")
            dist = distdecorr()
            clust = SingleLinkage()
            alloc = Naiverp()
            
            portafolio_hrp = HRPStyle(df, dist, clust, alloc)
            print(" Construyendo portfolio HRP...")
            pesos_hrp = portafolio_hrp.construir_hrp(start_train_date, end_train_date, start_bt_date, end_bt_date)
            
            print(f" Pesos HRP calculados: {len(pesos_hrp)} activos")
            portafolio_hrp.mostrar_pesos(pesos_hrp)
            
            print(" Generando gráfico de pastel...")
            portafolio_hrp.pastel(pesos_hrp)
            
            print(" Generando gráfico de barras...")
            portafolio_hrp.barras(pesos_hrp)
            
            print(" Generando gráfico de burbujas...")
            portafolio_hrp.bubbleplot_matplotlib(pesos_hrp)
            
            print("Generando correlaciones y dendograma...")
            portafolio_hrp.correlacion_y_dendograma()
            
            print(" Portfolio HRP completado exitosamente")
            
        except Exception as e:
            print(f" Error en Portfolio HRP: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*50)
        print(" ¡TODOS LOS PORTFOLIOS COMPLETADOS EXITOSAMENTE!")
        print("="*50)
        print(" Todas las gráficas se han abierto en pestañas del navegador")
        print("Los archivos HTML se han guardado en la carpeta 'plots'")
        
    except Exception as e:
        print(f" Error general en main: {e}")
        import traceback
        traceback.print_exc()

# run main
if __name__ == "__main__":
    main()