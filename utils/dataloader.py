import yfinance as yf
import os
from datetime import timedelta, datetime, date
import pandas as pd
import numpy as np
import ta
from ta.trend import MACD, ADXIndicator
from scipy.signal import argrelextrema
from arch import arch_model
from config import target_long_threshold, target_long_lookahead


def get_data(
        ticker, 
        start_date='1999-01-01', 
        end_date=None, 
        save_csv=False
):
    os.makedirs("data", exist_ok=True)
    filename = f"data/{ticker.lower()}.csv"
    # if os.path.exists(filename):
    #     print(f"Data for {i} already exists, skipping download.")
    #     continue

    # add one day to end_date because yf.download end is not inclusive
    if end_date is not None:
        end_date = pd.to_datetime(end_date).date()
        end_date = (end_date + timedelta(days=1)).isoformat()

    data = yf.download(
        ticker.upper(), 
        start=start_date, 
        end=end_date, 
        auto_adjust=True,
        progress=False
    )

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
def create_target_long(df, threshold=target_long_threshold, lookahead=target_long_lookahead):
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
    df['target_long'] = (df['future_return'] >= threshold).where(df['future_return'].notna())
    df.drop(columns=['seq_index', 'price_tmrw', 'future_max', 'future_return'], inplace=True)
    df.dropna(inplace=True)
    return df


def feature_engineering(df): # for trading at open tomorrow
    # bollinger bands and rsi
    df['sma30'] = df['close'].rolling(30).mean() # wont be used 
    df['sma10'] = df['close'].rolling(10).mean() # wont be used
    df['sma_diff'] = df['sma10'] - df['sma30']
    df['sma_slope'] = df['sma10'].diff()
    df['std30'] = df['close'].rolling(30).std() # wont be used 
    df['bollinger_upper'] = df['sma30'] + 2 * df['std30'] # wont be used 
    df['bollinger_lower'] = df['sma30'] - 2 * df['std30'] # wont be used 
    df['percent_b'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
    df['bollinger_z'] = (df['close'] - df['sma30']) / df['std30']
    df['price_near_lower_bb'] = (df['close'] <= df['bollinger_lower'] * 1.01).astype(int)
    df['rsi14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()    
    df['prod_bollingerz_rsi'] = df['percent_b'] * df['rsi14']

    # Detect local lows
    df['rsi_smooth'] = df['rsi14'].rolling(3).mean() # wont be used 
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

    # trend following contextual features
    # sma_slope (already added)
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    # df['macd'] = macd.macd()                   # EMA12 - EMA26
    # df['macd_signal'] = macd.macd_signal()     # 9-day EMA of MACD
    df['macd_diff'] = macd.macd_diff()         # Histogram: MACD - Signal

    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)

    df['adx'] = adx.adx()              # Trend strength
    df['adx_pos'] = adx.adx_pos()      # +DI; wont be used
    df['adx_neg'] = adx.adx_neg()      # -DI; wont be used

    # df['macd_uptrend'] = (df['macd_diff'] > 0).astype(int)
    df['strong_trend'] = (df['adx'] > 25).astype(int)
    df['up_trend_context'] = ((df['adx'] > 25) & (df['adx_pos'] > df['adx_neg'])).astype(int)
    df['down_trend_context'] = ((df['adx'] > 25) & (df['adx_neg'] > df['adx_pos'])).astype(int)

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
