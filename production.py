from config import drop_feats, backtest_config, watchlist
from utils.dataloader import (
    get_data, 
    create_target_long, 
    feature_engineering, 
    feature_engineering_add_macro
)
from utils.strategy import (
    train_rscv,
    train_with_best_param,
    backtest,
    get_cagr,
    get_sharpe
)

import os
import pickle
import logging # for future logging on cloud
from datetime import date, datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split

# -----

def load_dfs(filepath='dfs_storage.pkl'):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            dfs = pickle.load(f)
        return dfs
    else:
        return {}
    

def save_dfs(dfs, filepath='dfs_storage.pkl'):
    with open(filepath, 'wb') as f:
        pickle.dump(dfs, f)


# -----

def build_dict(ticker, end_date = None):
    try: 
        df = get_data(ticker, end_date=end_date)
        if not df.empty:
            my_d = {
                'ohlcv': df.copy(),
                'data_as_of': df.index.max().date().isoformat()
            }
        else:
            print(f"{ticker} returned empty data.")
            return None
    except Exception as e:
        print(f"{ticker} failed with: {e}")
        return None

    # data wrangling

    df = create_target_long(df)
    df = feature_engineering(df)
    df = feature_engineering_add_macro(df, 'spy')
    df = feature_engineering_add_macro(df, 'qqq')
    df.dropna(inplace=True)
    df_date_max = df.index.max().date().isoformat()

    X = df.drop(columns = ['target_long','ticker'])
    y = df['target_long']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    my_d['prepped_data'] = {
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

    # train
    print('training rscv...')
    fitted_train = train_rscv(X_train, y_train, drop_feats, n_iter=30)
    my_d['train_set_model'] = {
        'model': fitted_train,
        'best_estimator': fitted_train.best_estimator_,
        'best_params': fitted_train.best_params_,
        'best_score': fitted_train.best_score_,
        # 'trained_at': datetime.today().isoformat(),
    }
    print('rscv trained, best params found. now backtesting...')
    # pred and backtest

    df_trade, trade_stats = backtest(X_test, fitted_train, **backtest_config)
    total_trading_days = trade_stats['total_trading_days']

    my_d['test_set_result'] = {
        'backtest_config': backtest_config,
        **trade_stats,
        'cagr': get_cagr(df_trade),
        'sharpe': get_sharpe(df_trade, total_trading_days)
    }
    print('backtesting done. now train the final model...')
    # train the final model on all data
    final_model = train_with_best_param(X, y, drop_feats, fitted_train)
    my_d['final_model'] = final_model
    my_d['final_model_data_up_to'] = df_date_max
    my_d['last_update'] = datetime.today()

    print("\n" + "="*30)
    print(f"🐣{ticker} model ready LFG🗣️🗣️🗣️")
    print("="*30 + "\n")

    return my_d

# -----

def update_dfs(
        watchlist=watchlist, 
        tickers_to_retrain=None, 
        end_date = None,
        load_existing=True,
        force_retrain=False,
):
    if load_existing:
        dfs = load_dfs()
    else:
        dfs = {}

    tickers_to_retrain = tickers_to_retrain or []
    tickers_trained_this_time = []

    for idx, ticker in enumerate(watchlist):
        print(f"{ticker} out of {idx}/{len(watchlist)} tickers...")
        if ticker in tickers_to_retrain:  # should train
            print(f"{ticker} explicitly retraining...")
            result = build_dict(ticker, end_date)
            if result:
                dfs[ticker] = result
                tickers_trained_this_time.append(ticker)
            else:
                print(f"{ticker} skipped due to data issue.")
        elif ticker in dfs and not force_retrain:
            print(f"{ticker} skipped (last updated at {dfs[ticker]['last_update']}).")
            continue
        else:  # should train
            print(f"{ticker} training (new or forced)...")
            result = build_dict(ticker, end_date)
            if result:
                dfs[ticker] = result
                tickers_trained_this_time.append(ticker)
            else:
                print(f"{ticker} skipped due to data issue.")

    dfs['metadata'] = {
        'num_tickers': sum(1 for k in dfs if k != 'metadata'),
        'last_params': {
            'watchlist': watchlist,
            'end_date': end_date or 'default',
            'force_retrain': force_retrain,
            'tickers_to_retrain': tickers_to_retrain,
        },
        'last_updated_tickers': tickers_trained_this_time, 
        'last_updated_at': datetime.today().isoformat(),
    }

    save_dfs(dfs)
    return dfs


# -----

def run_pred(ticker, model = None, start_date = None, end_date = None): 
    if model is None: 
        dfs = load_dfs() 
        model = dfs[ticker]['final_model']
        model_date = dfs[ticker]['final_model_fit_with_data_up_to']
    else: 
        model_date = None

    if start_date is None:
        year_ahead = date.today() - timedelta(days=365)
    else:
        year_ahead = datetime.strptime(start_date, "%Y-%m-%d").date() - timedelta(days=365)

    # get at least a year of data; get_data also handles end_date to be inclusive already
    data = get_data(ticker, start_date = year_ahead, end_date = end_date)
    if data is None or len(data)==0:
        print('no data queried from yfinance')
        return 0
    
    most_recent_date = data.index.max().date().isoformat()

    data = feature_engineering(data)
    data = feature_engineering_add_macro(data, 'spy')
    data = feature_engineering_add_macro(data, 'qqq')

    start = start_date or most_recent_date
    end = end_date or most_recent_date

    data_to_pred = data[start:end]

    if model_date is not None and data_to_pred.index.min() <= pd.Timestamp(model_date):
        print(f"WARNING: this model (uses data up to {model_date.isoformat()})"
              f"should not be used to predict data from {data_to_pred.index().min().date().isoformat()}"
              f"to {model_date.isoformat()}")
        return 0
    elif model_date is None: 
        print("WARNING: make sure model was not trained on any data using for prediction here!")
    
    if data_to_pred.isna().any().any(): 
        print('WARNING: NaN value detected in data')

    y_hardpred = model.predict(data_to_pred)
    y_softpred = model.predict_proba(data_to_pred)[:,1]

    my_d = {
        'ticker': [ticker] * len(y_hardpred),
        'pred': y_hardpred, # should enter the next day?
        'proba': y_softpred,
        'sma30': data_to_pred['sma30'], # sma30 as of the date
        'atr': data_to_pred['atr'], # atr as of the date
        'garch_vol': data_to_pred['garch_vol'] # garch volatility as of the date 
    }

    return pd.DataFrame(my_d, index=data_to_pred.index)