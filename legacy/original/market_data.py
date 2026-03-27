import numpy as np 
import yfinance as yf 
import pandas as pd
import time
import geopandas as gdp
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="geo_stock_map")


class MarketData:
    def __init__(self, tickers):
        self.tickers = tickers
        self.geolocator = Nominatim(user_agent="portfolio_app")
        self.historical_data = None
        self.asset_info = None
        self.download_data()



    def download_data(self):
        data = {}
        info_list = []
        for ticker in self.tickers:
            #descargar los datos 
            tkr = yf.Ticker(ticker) #declarar el ticker
            df = tkr.history(period="5y") #obtener la información historica 
            #volverlo un solo df con multiindex 
            df.drop(columns=["Dividends", "Stock Splits"], errors='ignore', inplace=True)  #no relevantes en el de ahorita quiza madure en el futuro y lo incluya
            df.columns = pd.MultiIndex.from_product([[ticker], df.columns])  #crear el multiindex de mi df solito
            data[ticker] = df #guardarlo
            #el df de la información 
            info = tkr.info
            #los datos que me interesan
                #geocodificación para mapas haha 
            city = info.get('city', '')
            state = info.get('state', '')
            country = info.get('country', '')
            address = f"{city}, {state}, {country}"
            
            
            latitude, longitude = None, None
            try:
                location = geolocator.geocode(address)
                if location:
                    latitude = location.latitude
                    longitude = location.longitude
            except Exception as e:
                print(f"Error geocoding {ticker}: {e}")
            info_data = {
            'Ticker': ticker,
            'Name': info.get('longName'),
            'Sector': info.get('sector'),
            'Industry': info.get('industry'),
            'City': city,
            'State': state,
            'Country': country,
            'Exchange': info.get('exchange'),
            'Currency': info.get('currency'),
            'Address': address,
            'Latitude': latitude,
            'Longitude': longitude
            }

            info_list.append(info_data)
            time.sleep(1)  # para que yahoo no me cancele otra vez
        #hacer el merge solo una vez de los datos historicos 
        self.historical_data = pd.concat(data.values(), axis=1)
        
        #volver mi larga lista de atributos un solo df 
        
        self.asset_info = pd.DataFrame(info_list).set_index('Ticker')

        return self.historical_data, self.asset_info
  