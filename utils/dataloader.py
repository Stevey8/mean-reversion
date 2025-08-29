import yfinance as yf
import os
import pickle
import pytz
from tqdm import tqdm
from datetime import timedelta, datetime, date
import pandas as pd
import numpy as np
import ta
import urllib.request
from typing import Iterable, List, Set, Optional
from ta.trend import MACD, ADXIndicator
from scipy.signal import argrelextrema
from arch import arch_model


# ===============================
# ===== serializing objects =====
# ===============================
def save_pkl(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved data to {path}.")

def read_pkl(path, redo=False): # only for dictionary
    if not os.path.exists(path):
        print(f"{path} does not exist, starting fresh.")
        return {}
    elif redo: 
        print('redo set to True, starting fresh.')
        return {}
    with open(path, 'rb') as f:
        dic = pickle.load(f)
    print(f"Loaded data from {path}.")
    return dic

# =======================
# ===== data loader =====
# =======================


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
        # print(f"No data found for {ticker}")
        return data
    
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


# get list of tickers in s&p500
def get_sp500(
    path: str = "files/sp500_tickers.pkl",
    redo: bool = False,
    *,
    url: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    excluded: Iterable[str] = ("GOOG", "FOXA", "NWS", "WBA"),  # adjust as needed
    lowercase: bool = True,
    cache_min_count: int = 490,  # accept cached list if it's reasonably complete
) -> List[str]:
    """
    Return a list of S&P 500 tickers suitable for Yahoo/Polygon-style symbols
    (i.e., dots replaced with dashes). Caches to a pickle via `save_pkl`/`read_pkl`.

    Notes
    -----
    - Wikipedia can change; this scrapes the first table with a 'Symbol' column.
    - Some providers use '-' instead of '.' for class shares (e.g., BRK.B -> BRK-B).
    - `excluded` lets you drop alternates (e.g., GOOG vs GOOGL) if desired.
    - 2025-08-25: IBKR replaced WBA 
    """

    # Try cache first unless forced refresh
    try:
        cached = read_pkl(path, redo=False)
        if isinstance(cached, list) and not redo and len(cached) >= cache_min_count:
            return cached
    except Exception:
        # Ignore cache errors; we’ll try to fetch fresh
        pass

    # Build a polite opener (helps avoid occasional 403s from Wikipedia)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; sp500-fetch/1.0)"},
        )
        with urllib.request.urlopen(req) as resp:
            html = resp.read()
        # Parse tables from the downloaded HTML to avoid a second network call
        tables = pd.read_html(html, header=0)
    except Exception as e:
        # On fetch/parsing error, fall back to cache if available
        try:
            cached = read_pkl(path, redo=False)
            if isinstance(cached, list) and len(cached) > 0:
                return cached
        except Exception:
            pass
        raise RuntimeError(f"Failed to fetch/parse S&P 500 table: {e}")

    # Find the table containing a 'Symbol' column (usually the first table)
    sp500_df: Optional[pd.DataFrame] = None
    for tbl in tables:
        if "Symbol" in tbl.columns:
            sp500_df = tbl
            break
    if sp500_df is None:
        raise RuntimeError("Could not find a table with a 'Symbol' column.")

    # Extract and normalize tickers
    syms = sp500_df["Symbol"].astype(str).str.strip().tolist()

    # Replace '.' with '-' for data provider compatibility (e.g., BRK.B -> BRK-B)
    syms = [s.replace(".", "-") for s in syms]

    # Exclude specified symbols (match against uppercase before final casing)
    excluded_set: Set[str] = {s.upper() for s in excluded}
    syms = [s for s in syms if s.upper() not in excluded_set]

    # Optional casing
    if lowercase:
        syms = [s.lower() for s in syms]

    # De-duplicate and sort for deterministic output
    unique_sorted = sorted(set(syms))

    # Save and return
    save_pkl(path, unique_sorted)
    return unique_sorted



# get metadata
def get_metadata(ticker, sp500_tickers=None):
    try:
        info = yf.Ticker(ticker.upper()).info
        if sp500_tickers is None: 
            sp500_tickers = get_sp500()
        return {
            'ticker': ticker,
            'sector': info.get('sector'),
            # 'industry': info.get('industry'), 
            # # try not to ohe this for now because its a lot more granular
            'market_cap': info.get('marketCap'),
            'avg_volume': info.get('averageVolume'),
            'beta': info.get('beta') if info.get('beta') is not None else 1,
            'is_sp500': ticker in sp500_tickers
        }
    except Exception as e:
        print(f"Failed for {ticker}: {e}")
        return None


def get_sector_mindate(ticker: str):
    """
    Returns dict {'ticker','sector','min_date'} or None on failure.
    min_date is a python date (not datetime) for compactness.
    """
    try:
        tkr = yf.Ticker(ticker.upper())
        sector = tkr.info.get('sector', None)  # ok if None
        ohlcv = yf.download(ticker.upper(), period='max', progress=False, auto_adjust=False)
        if ohlcv is None or ohlcv.empty:
            raise ValueError("No OHLCV returned")
        min_dt = ohlcv.index.min()
        if pd.isna(min_dt):
            raise ValueError("min index is NaT")
        return {
            'ticker': ticker.lower(),
            'sector': sector,
            'min_date': pd.to_datetime(min_dt).date()
        }
    except Exception as e:
        print(f"Failed for {ticker}: {e}")
        return None


def get_sp500_tickers_sectors_mindates(
    path: str = 'files/df_sp500_tickers_sectors_mindates.pkl',
    sp500_tickers: list[str] | None = None,
    redo: bool = False
) -> pd.DataFrame:
    """
    Build (or load) a DataFrame with columns ['ticker','sector','min_date'].
    """
    if os.path.exists(path) and not redo:
        return read_pkl(path)

    if sp500_tickers is None:
        sp500_tickers = get_sp500()

    rows = []
    for tk in tqdm(sp500_tickers, desc="Processing tickers for get_sp500_tickers_sectors_mindates()"):
        rec = get_sector_mindate(tk)
        if rec is not None:
            rows.append(rec)

    df = pd.DataFrame(rows)
    # normalize types
    if not df.empty:
        df['min_date'] = pd.to_datetime(df['min_date']).dt.date

    save_pkl(path, df)
    return df


def select_global_candidates(
    info: pd.DataFrame | None = None,
    min_date: str = '2016-01-01',
    n: int = 10,  # sample n tickers per sector
    path: str = 'files/df_global_candidates.pkl',
    redo: bool = False,
) -> pd.DataFrame:
    """
    From an info DataFrame (ticker, sector, min_date), mark:
      - valid_candidate: ticker's min_date < cutoff
      - selected: sampled True/False per sector among valid candidates (n per sector)
    Saves the resulting DataFrame to `path` and returns it.
    """
    if os.path.exists(path) and not redo:
        return read_pkl(path)

    # Load info if not provided
    print("select_global_candidates starting fresh")
    if info is None:
        info = get_sp500_tickers_sectors_mindates(redo=False)

    df = info.copy()
    if df.empty:
        # Nothing to do
        print("df is empty; stored this empty df as pkl.")
        save_pkl(path, df)
        return df

    # Normalize types
    # Convert min_date column to pandas datetime (drop tz) then to date for clear comparison
    df['min_date'] = pd.to_datetime(df['min_date']).dt.date
    cutoff = pd.to_datetime(min_date).date()

    # 1) valid_candidate
    df['valid_candidate'] = df['min_date'] < cutoff

    # Guard against sectors being missing; replace None with 'Unknown' (optional)
    df['sector'] = df['sector'].fillna('Unknown')

    # 2) Sample n per sector from valid candidates
    valid = df[df['valid_candidate']].copy()

    def _sample_group(g):
        # If group smaller than n, take all
        m = min(n, len(g))
        return g.sample(n=m) if m > 0 else g.iloc[0:0]

    if valid.empty:
        sampled_index = pd.Index([])
    else:
        sampled_index = (
            valid.groupby('sector', group_keys=False)
                 .apply(_sample_group)
                 .index
        )

    # 3) selected flag
    df['selected'] = False
    df.loc[sampled_index, 'selected'] = True

    # Save & return
    save_pkl(path, df)
    return df





# =======================
# ===== feature eng =====
# =======================


def feature_engineering(df): # for trading at open tomorrow
    df = df.copy()
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
    df['rolling_volatility14'] = df['daily_return'].rolling(window=14).std()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()

    # GARCH(1,1) on returns
    returns = df['open'].pct_change().dropna() * 100 # in percent
    am = arch_model(returns, vol='Garch', p=1, q=1, mean='constant')
    res = am.fit(disp='off')
    df['garch_vol'] = res.conditional_volatility # in percent

    # time related features
    # df['year'] = df.index.year
    year_min = df.index.year.min()
    df['year_since'] = df.index.year-year_min
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


# useful for adding information from market trend e.g. spy, qqq
def market_trend(etf:str, save_csv = False):
    # always get the most up-to-date data (no need to set end_date)
    df_etf = get_data(etf)

    df_etf['sma10'] = df_etf['close'].rolling(10).mean()
    df_etf['sma20'] = df_etf['close'].rolling(20).mean()
    df_etf['sma50'] = df_etf['close'].rolling(50).mean()
    df_etf['sma200'] = df_etf['close'].rolling(200).mean()
    df_etf[f'{etf}_trend_10_50'] = (df_etf['sma10'] > df_etf['sma50']).astype(int)
    df_etf[f'{etf}_trend_20_50'] = (df_etf['sma20'] > df_etf['sma50']).astype(int)
    df_etf[f'{etf}_trend_50_200'] = (df_etf['sma50'] > df_etf['sma200']).astype(int)
    df = df_etf[[f'{etf}_trend_10_50',f'{etf}_trend_20_50',f'{etf}_trend_50_200']]
    if save_csv:
        csv_path = f"data/{etf}.csv"
        df.to_csv(csv_path)
    return df


def make_market_trend_dict(list_of_etfs = ['spy','qqq']):
    d = {}
    for i in list_of_etfs:
        d[i] = market_trend(i)
    return d
    

def feature_engineering_market(df, etfs = None): 
    df = df.copy()
    if etfs is None: 
        etfs = make_market_trend_dict()
    for v in etfs.values(): 
        df = df.merge(v, left_index=True, right_index=True, how='left')
    return df
        

def feature_engineering_metadata(df, sp500_tickers):
    df = df.copy()
    if 'ticker' not in df.columns:
        raise KeyError("df does not contain ticker info")
    if df['ticker'].nunique() != 1: 
        raise ValueError(f"Expected exactly 1 unique ticker, but got {df['ticker'].nunique()}")
    ticker = df['ticker'].iloc[0]
    md = get_metadata(ticker, sp500_tickers=sp500_tickers)
    for k, v in md.items():
        df[k] = v 
    return df

# ===========================
# ===== target creation =====
# ===========================

# trading at open tomorrow (t+1)
# based on today's data (ohlcv and more), decide whether to long tomorrow
def create_target_long(df_orig, lookahead=5, timing = 'open', strategy='static', **kwargs) -> pd.DataFrame:
    """
    entry: assume always enter at open
    exit: timing is "exit" timing, can either be 'open' or 'high'
    use 'high' when your platform performs auto exit for you when price exceeds your profit threshold 
    use 'open' for simplification (only trade at open)

    strategy: either 'static' or 'dynamic'
    static: use a set (expected return rate) threshold throughout, default is 0.01
    kwargs for static: threshold 
    dynamic: mimic the entry_exit function below, use a profit loss ratio scaled by volatility
    kwargs: vol (one of 'atr', 'garch_vol'), upper = 0.95, take_profit (default 1.2)
    upper is the quantile that cap the volatility (so that won't have extremely high threshold caused by high volatility)
    (no need stop loss here; everything not over profit threshold will be labeled 0)

    label as 1 if max return over the next 'lookahead' days is >= threshold, else label as 0.
    only returns a series (ie y), not a dataframe
    """
    df = df_orig.copy()

    if timing not in ['open','high']: 
        raise ValueError(f"Invalid timing value: {timing}. Expected 'open' or 'high'.") 
    
    if strategy == 'static':
        allowed = {'threshold'}
    elif strategy == 'dynamic':
        allowed = {'vol', 'upper', 'take_profit', 'min_return'}
        # vol: either atr (in dollars) or garch volatility (in percent)
        # upper: use up to this percentile of the volatility 
        # take_profit: the multiplier of vol
        # min_return: the lower threshold in % for return 

    else: 
        raise ValueError(f"Invalid strategy value: {strategy}. Expected 'static' or 'dynamic'.") 

    for k in kwargs:
        if k not in allowed:
            raise ValueError(f"Unexpected keyword argument: '{k}' for strategy='{strategy}'")


    df['price_tmrw'] = df['open'].shift(-1) # if 'enter' tomorrow (t+1) at open

    df['seq_index'] = range(len(df))
    future_max_list = []
    for i in df['seq_index']:
        if i + 1 + lookahead <= len(df):
            window = df[timing].iloc[i : i+1+lookahead] # try exit at timing (open/high) from t+1 to t+5
            future_max_list.append(window.max())
        else:
            future_max_list.append(np.nan)

    df['future_max'] = future_max_list
    df['future_return'] = (df['future_max'] - df['price_tmrw']) / df['price_tmrw'] 

    if strategy == 'static':
        threshold = kwargs.get('threshold', 0.01)
        df['threshold'] = threshold 

    elif strategy == 'dynamic': 
        vol = kwargs.get('vol')
        if vol not in df.columns:
            raise KeyError(f"Volatility column '{vol}' not found in DataFrame")
        
        upper = kwargs.get('upper', 0.95)
        take_profit = kwargs.get('take_profit', 1)
        min_return = kwargs.get('min_return', 0.01)

        if vol=='atr': 
            df['atr_pct'] = df['atr']/df['price_tmrw'] 
            vol_cap = df['atr_pct'].quantile(upper)
            df['effective_vol'] = df['atr_pct'].clip(upper=vol_cap, lower = min_return)
        elif vol=='garch_vol':
            vol_cap = df['garch_vol'].quantile(upper)
            df['effective_vol'] = df['garch_vol'].clip(upper=vol_cap, lower = min_return*100) / 100
        else:   
            raise ValueError("Invalid or missing 'vol'. Must be one of: 'atr', 'garch_vol'")
        
        df['threshold'] = df['effective_vol']*take_profit
    
    else: 
        raise ValueError(f"Invalid strategy value: {strategy}. Expected 'static' or 'dynamic'.") 
    
    df['target_long'] = (df['future_return'] >= df['threshold']) \
        .where(df['future_return'].notna()) \
        .astype('Int64')  

    # Do not drop the last five rows with NaN in 'target_long'; keep them for correct alignment
    return df_orig.merge(df[['target_long']], how='left', left_index=True, right_index=True)


# ===================================================
# ===== wrangling (for all features and target) =====
# ===================================================
def wrangling(
        df_ohlcv, 
        etfs=None, # if creating a list of tickers, should load the etfs (dict) prior to this and assign here 
        sp500_tickers=None,
        **kwargs
):
    df = feature_engineering(df_ohlcv)
    df = feature_engineering_market(df, etfs)
    df = feature_engineering_metadata(df, sp500_tickers=sp500_tickers)
    # drop na first just to be safe for running target creation 
    df.dropna(inplace=True)

    # should not drop na (keep the last five rows)
    df = create_target_long(df, **kwargs) 
    return df



# ==========

# # when querying new data everyday, only generate data for this day(s) only
# def feature_engineering_incremental(df_all, df_new):
#     # Keep buffer for rolling windows
#     buffer = 30  
#     df_slice = pd.concat([df_all.tail(buffer), df_new])
#     df_slice = feature_engineering(df_slice)
#     return pd.concat([df_all.iloc[:-buffer], df_slice])
