import yfinance as yf
import os
import pandas as pd
import numpy as np
import ta
from scipy.signal import argrelextrema


def get_data(ticker, start_date='1999-01-01', end_date=None, save_csv=False):
    os.makedirs("data", exist_ok=True)
    filename = f"data/{ticker.lower()}.csv"
    # if os.path.exists(filename):
    #     print(f"Data for {i} already exists, skipping download.")
    #     continue

    data = yf.download(ticker.upper(), start=start_date, end=end_date, auto_adjust=True)

    # Handle empty or failed download
    if data.empty:
        print(f"No data found for {ticker}")
        return pd.DataFrame()
    
    # Flatten column headers if it's a MultiIndex (e.g., from group_by='ticker')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data['Ticker'] = ticker.lower()
    data.index = pd.to_datetime(data.index)
    data.columns = [col.lower() for col in data.columns]


    if save_csv:
        data.to_csv(filename, index=False)
        print(f"Saved data for {ticker} to {filename}")
        
    return data


# model for longing; trading at open tomorrow (t+1)
# based on today's data (ohlcv and more), decide whether to long tomorrow
# binary target: 1 if price reverts up (above the threshold e.g. gain over 2%) wihtin the next five trading days, 0 otherwise
def create_target_long(df, threshold=0.02, lookahead=5):
    """
    Label as 1 if max return over the next 'lookahead' days is >= threshold.
    Else, label as 0.
    """
    # df = df.reset_index().rename(columns={'Date': 'date'})
    df['price_tmrw'] = df['open'].shift(-1) # trade at open next day

    df['seq_index'] = range(len(df))
    future_max_list = []
    for i in df['seq_index']:
        if i + 1 + lookahead <= len(df):
            window = df['open'].iloc[i+1 : i+1+lookahead]
            future_max_list.append(window.max())
        else:
            future_max_list.append(np.nan)

    df['future_max'] = future_max_list

    df['future_return'] = (df['future_max'] - df['price_tmrw']) / df['price_tmrw']
    df['target_long'] = (df['future_return'] >= threshold).astype(int)
    df.drop(columns=['seq_index', 'price_tmrw', 'future_max', 'future_return'], inplace=True)
    return df


def feature_engineering(df): # for trading at open tomorrow
    # bollinger bands and rsi
    df['sma30'] = df['close'].rolling(30).mean()
    df['sma10'] = df['close'].rolling(10).mean()
    df['sma_diff'] = df['sma10'] - df['sma30']
    df['sma_slope'] = df['sma10'].diff()
    df['std30'] = df['close'].rolling(30).std()
    df['bollinger_upper'] = df['sma30'] + 2 * df['std30']
    df['bollinger_lower'] = df['sma30'] - 2 * df['std30']
    df['percent_b'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
    df['bollinger_z'] = (df['close'] - df['sma30']) / df['std30']
    df['price_near_lower_bb'] = (df['close'] <= df['bollinger_lower'] * 1.01).astype(int)
    df['rsi14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()    
    df['prod_bollingerz_rsi'] = df['percent_b'] * df['rsi14']

    # Detect local lows
    df['rsi_smooth'] = df['rsi14'].rolling(3).mean()
    rsi_vals = df['rsi_smooth'].values
    local_lows = argrelextrema(rsi_vals, np.less, order=5)[0]
    df['rsi_local_low'] = 0
    df.iloc[local_lows, df.columns.get_loc('rsi_local_low')] = 1

    # some other useful features  
    df['daily_return'] = df['open'].pct_change()
    df['rolling_volatility14'] = df['daily_return'].rolling(window=30).std()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()

    # GARCH(1,1) on returns
    returns = df['close'].pct_change().dropna() * 100 # in percent
    am = arch_model(returns, vol='Garch', p=1, q=1)
    res = am.fit(disp='off')
    df['garch_vol'] = res.conditional_volatility # in percent

    # time related features
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['week'] = df.index.isocalendar().week
    df['dayofweek'] = df.index.dayofweek

    # df.dropna(inplace=True)
    return df


# useful for adding information from macro market trend
# e.g. spy, qqq
def feature_engineering_add_macro(df,etf:str):
    df_etf = get_data(etf, end_date = df.index.max())
    df_etf['sma10'] = df_etf['close'].rolling(10).mean()
    df_etf['sma20'] = df_etf['close'].rolling(20).mean()
    df_etf['sma50'] = df_etf['close'].rolling(50).mean()
    df_etf['sma200'] = df_etf['close'].rolling(200).mean()
    df_etf['trend_10_50'] = (df_etf['sma10'] > df_etf['sma50']).astype(int)
    df_etf['trend_20_50'] = (df_etf['sma20'] > df_etf['sma50']).astype(int)
    df_etf['trend_50_200'] = (df_etf['sma50'] > df_etf['sma200']).astype(int)
    df = df.merge(df_etf['trend_10_50'].rename(f'{etf}_trend_10_50'), left_index=True, right_index=True, how='left')
    df = df.merge(df_etf['trend_20_50'].rename(f'{etf}_trend_20_50'), left_index=True, right_index=True, how='left')
    df = df.merge(df_etf['trend_50_200'].rename(f'{etf}_trend_50_200'), left_index=True, right_index=True, how='left')
    return df
