import pandas as pd
import numpy as np

import requests
from io import StringIO

from datetime import datetime

import sqlite3

def try_to_transform_data_to_numeric(column_data):
    try:
        return column_data.str.replace('.', '', regex=False).str.replace(',', '.').astype(float)
    except:
        return column_data
    
def get_market_info():
    try:
        url_burza = f'https://rest.zse.hr/web/Bvt9fe2peQ7pwpyYqODM/price-list/XZAG/{datetime.now().strftime("%Y-%m-%d")}/csv?language=HR'
        response = requests.get(url_burza)
        data = StringIO(response.text)
        market_info = pd.read_csv(data, delimiter=';')
        market_info = pd.read_parquet('market_info.parquet')
#         if len(market_info) > 0:
#             market_info.to_parquet('market_info.parquet')
        return market_info
    except:
        return pd.read_parquet('market_info.parquet')
    
def get_one_raw_stock_data(stock, market_info=None, start_time='2023-01-01', end_time=datetime.now().strftime("%Y-%m-%d")):
    if market_info is None:
        market_info = get_market_info()
    
    full_code = market_info.query(f'symbol == "{stock}"').iloc[0]['isin']
    url = f'https://rest.zse.hr/web/Bvt9fe2peQ7pwpyYqODM/security-history/XZAG/{full_code}/{start_time}/{end_time}/csv?language=HR'
    response = requests.get(url)

    data = StringIO(response.text)
    return pd.read_csv(data, delimiter=';')

def get_one_stock_data(stock, market_info=None, start_time='2023-01-01', end_time=datetime.now().strftime("%Y-%m-%d"), columns_to_return='returns'):
    df = get_one_raw_stock_data(stock, market_info=market_info, start_time=start_time, end_time=end_time)
    
    df = df[df['trading_model_id'] == 'CT']
    
    if 'last_price' in df.columns:
        df['open_price'] = try_to_transform_data_to_numeric(df['open_price'])
        df['high_price'] = try_to_transform_data_to_numeric(df['high_price'])
        df['low_price']  = try_to_transform_data_to_numeric(df['low_price'])
        df['last_price'] = try_to_transform_data_to_numeric(df['last_price'])
        df['vwap_price'] = try_to_transform_data_to_numeric(df['vwap_price'])
        df['turnover']   = try_to_transform_data_to_numeric(df['turnover'])
        df['change_prev_close_percentage']   = try_to_transform_data_to_numeric(df['change_prev_close_percentage']) / 100

        df = df.set_index('date')
        if 'HRK' in df['price_currency'].values:
            df.loc[df['price_currency'] == 'HRK', ['open_price', 'high_price', 'low_price', 'last_price', 'vwap_price', 'turnover']] /= 7.53450
            df['price_currency'] == 'EUR'
            df['turnover_currency'] == 'EUR'
    
    df.index = pd.to_datetime(df.index)
    df = df.resample('1D').asfreq()
    
    df['change_prev_close_percentage'] = df['change_prev_close_percentage'].fillna(0)
    
    if columns_to_return == 'returns':
        return df['change_prev_close_percentage'].rename(stock)
        
    elif columns_to_return == 'prices':
        df = df['last_price'].rename(stock)

    if columns_to_return == 'prices_and_returns':
        df = df[['last_price', 'change_prev_close_percentage']]
    
    return df.fillna(method='ffill')

def get_stock_prices(stocks, start_time='2020-01-01', end_time=datetime.now().strftime("%Y-%m-%d")):
    burza_info = get_market_info()
    
    df_dict = {}
    response_dict = {}

    for dionica in stocks:
        try:
            df = get_one_stock_data(dionica, burza_info, start_time=start_time, end_time=end_time, columns_to_return='returns')

            df_dict[dionica] = df
            print(f'{dionica} - uspješno dohvaćeno')
        except Exception as e:
            print(f'{dionica} - neuspjeh: {e}')
            
    return pd.concat(df_dict.values(), axis=1).resample('1D').asfreq().fillna(method='ffill')

def get_stock_prices_and_returns(stocks, start_time='2020-01-01', end_time=datetime.now().strftime("%Y-%m-%d")):
    burza_info = get_market_info()
    
    df_price_dict = {}
    df_returns_dict = {}
    response_dict = {}

    for dionica in stocks:
        try:
            df = get_one_stock_data(dionica, burza_info, start_time=start_time, end_time=end_time, columns_to_return='baza_podataka')

            df_price_dict[dionica] = df['last_price'].rename(dionica)
            df_returns_dict[dionica] = df['change_prev_close_percentage'].rename(dionica)
            print(f'{dionica} - uspješno dohvaćeno')
        except Exception as e:
            print(f'{dionica} - neuspjeh: {e}')

    prices_df = pd.concat(df_price_dict.values(), axis=1)
    prices_df.index = pd.to_datetime(prices_df.index)
    
    returns_df = pd.concat(df_returns_dict.values(), axis=1)
    returns_df.index = pd.to_datetime(returns_df.index)

    return prices_df.resample('1D').asfreq().fillna(method='ffill'), returns_df.resample('1D').asfreq().fillna(0)


def get_stock_options():
    burza_info = get_market_info()

    stocks = list(burza_info['symbol'].unique())
    stocks.sort()
    # sorted(stocks)
    return stocks

def transform_data(df):
    df = df.copy()
    nan_columns =  list(df.columns[df.isnull().all()])
    df = df.drop(nan_columns, axis=1).copy()
    
    returns = df.pct_change().fillna(0)
    returns = df.fillna(0)
    
    zero_return_columns = list(returns.columns[(returns == 0).all()])
    
    return {
        'returns': returns.drop(zero_return_columns, axis = 1),
        'nan_columns': nan_columns,
        'zero_return_columns': zero_return_columns
    }

############################################# BAZA #############################################

def fill_database(stocks=get_stock_options(), start_time='2015-01-01', end_time=datetime.now().strftime("%Y-%m-%d"), db_name='baza_podataka.db'):
    print('--- fill_database --- POCETAK')
    print(f'--- fill_database --- dohvacanje podataka od {start_time} do {end_time}')
    prices, returns_data = get_stock_prices_and_returns(stocks, start_time, end_time)
    
    print('--- fill_database --- obrada podataka')

    prices = prices.reset_index()
    returns_data = returns_data.reset_index()

    if prices.columns[0] == 'index':
        prices.rename(columns={'index': 'date'}, inplace=True)

    if returns_data.columns[0] == 'index':
        returns_data.rename(columns={'index': 'date'}, inplace=True)
    print(prices)
    prices_melted = pd.melt(prices, id_vars=['date'], var_name='stock', value_name='prices')
    returns_melted = pd.melt(returns_data, id_vars=['date'], var_name='stock', value_name='returns')

    final_df = pd.merge(returns_melted, prices_melted, on=['date', 'stock'])
    
    print('--- fill_database --- spremanje podataka')
    conn = sqlite3.connect(db_name)

    final_df.to_sql('returns_and_prices', conn, index=False, if_exists='replace')

    conn.close()
    print('--- fill_database --- KRAJ')
    return 'Baza je napunjena'

def read_from_db(stocks=get_stock_options() , start_time='2015-01-01', end_time=None, db_name='baza_podataka.db', read_all=False):
    conn = sqlite3.connect(db_name)
    
    if stocks is None:
        return None, None
        
    if end_time is None:
        end_time = datetime.now().strftime("%Y-%m-%d")

    query = f"""
        SELECT date, stock, returns, prices
        FROM returns_and_prices
        WHERE stock IN ({','.join(['?' for _ in stocks])})
        AND date >= ?
        AND date <= ?
    """

    params = stocks + [start_time, end_time]
    
    df = pd.read_sql(query, conn, params=params)
    prices = df.pivot(columns='stock', index='date', values='prices')
    returns = df.pivot(columns='stock', index='date', values='returns')

    prices.index = pd.to_datetime(prices.index)
    returns.index = pd.to_datetime(returns.index)

    conn.close()
    
    return prices, returns


def get_last_date_from_db(db_name='baza_podataka.db'):
    try:
        conn = sqlite3.connect(db_name)
    
        query = f"""
            SELECT MAX(date) AS last_date
            FROM returns_and_prices
        """
        
        df = pd.read_sql(query, conn)
        
        conn.close()
        
        last_date = df['last_date'].iloc[0] 
        return last_date
    except:
        return '2015-01-01'