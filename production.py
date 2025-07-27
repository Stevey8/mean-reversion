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
from tabulate import tabulate


# -----

def load_dfs(filepath='dfs_storage.pkl', metadata=False):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            dfs = pickle.load(f)
        if metadata:
            print("NOTE: metadata included in this dfs")
            print("get metadata by calling one of its keys from dfs['metadata']:")
            print(dfs['metadata'].keys())
        else:
            dfs.pop('metadata', None)
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
    # trade states:
        # 'exit_reason_spread'
        # 'holding_days_spread'
        # 'total_trading_days'
        # 'total_holding_days'
        # 'holding_time_percentage'
        # 'n_trades'
        # 'n_wins'
        # 'win_rate'

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
        dfs = None,
        tickers_to_retrain=None, 
        end_date = None,
        load_existing=True,
        force_retrain=False,
):
    if load_existing:
        dfs = load_dfs()
    elif dfs is None: 
        dfs = {}

    tickers_to_retrain = tickers_to_retrain or []
    tickers_trained_this_time = []

    if any(
        x is not None and not isinstance(x, (list, tuple))
        for x in [watchlist, tickers_to_retrain]
    ):
        raise TypeError("watchlist and tickers_to_retrain must be list or tuple (or None).")

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

def run_pred(ticker, model = None, model_date = None, start_date = None, end_date = None): 
    if model is None: 
        dfs = load_dfs() 
        model = dfs[ticker]['final_model']
        model_date = dfs[ticker]['final_model_data_up_to']

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

    if start_date is None and end_date is None:
        data_to_pred = data[most_recent_date:most_recent_date]
    elif end_date is None: # has start date but no end date 
        data_to_pred = data[start_date:most_recent_date]
    elif start_date is None: # has end date but no start date 
        data_to_pred = data[end_date:end_date]
    else:
        data_to_pred = data[start_date:end_date]

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
        'date': pd.to_datetime(data_to_pred.index),
        'next_day_over_50%?': y_hardpred, # should enter the next day?
        'proba': y_softpred,
        'sma30': data_to_pred['sma30'], # sma30 as of the date
        'atr': data_to_pred['atr'], # atr as of the date
        'garch_vol': data_to_pred['garch_vol'] # garch volatility as of the date 
    }

    return pd.DataFrame(my_d) # no (meaning default) index


def run_pred_dfs(
    tickers=watchlist,
    dfs=None,
    start_date=None, 
    end_date=None,
    load_existing=True
):
    if load_existing:
        dfs = load_dfs()
    elif dfs is None:
        raise ValueError("No data provided (dfs is None).")

    if not isinstance(tickers, (list, tuple)):
        raise TypeError("tickers must be a list or tuple.")

    df_all = []
    for t in tickers:
        if t not in dfs:
            print(f"{t} not found in dfs — skipping.")
            continue
        model = dfs[t]['final_model']
        model_date = dfs[t]['final_model_data_up_to']
        df_pred = run_pred(t, model, model_date, start_date, end_date)
        df_all.append(df_pred)

    if df_all:
        return pd.concat(df_all).set_index('ticker')
    else:
        return pd.DataFrame()


# -----

def check(
        dfs=None, 
        load_existing=True, 
        with_pred=True,
        print_df=True,
        save_csv=True
):
    print("checking...")
    if load_existing:
        dfs = load_dfs()
    elif dfs is None:
        print("no data available (dfs is None)")
        return None

    # ---- Collect backtest stats ----
    my_d = {}
    for k, v in dfs.items():
        my_d[k] = {
            'total_trading_days': v['test_set_result']['total_trading_days'],
            'holding_time_percentage': v['test_set_result']['holding_time_percentage'],
            'n_trades': v['test_set_result']['n_trades'],
            'win_rate': v['test_set_result']['win_rate'],
            'cagr': v['test_set_result']['cagr'],
            'sharpe': v['test_set_result']['sharpe'],
        }

    if len(my_d) == 0:
        print('empty dfs, no backtest result and prediction can be shown')
        return None
    
    df_stats = pd.DataFrame.from_dict(my_d, orient='index')
    df_stats.index.name = 'ticker'

    # ---- get prediction ----
    df_pred = None
    if with_pred: 
        df_pred = run_pred_dfs(dfs=dfs, load_existing=False) 
        max_date = df_pred['date'].max().date().isoformat()
        df_pred['backtest→'] = None
        df = df_pred.merge(df_stats, on='ticker', how='left')
    else: 
        df = df_stats
        max_date = None

    # ---- Save prediction DataFrame ----
    if with_pred and save_csv and df_pred is not None and not df_pred.empty:
        os.makedirs("predictions", exist_ok=True)
        df.to_csv(f"predictions/predictions_{max_date}.csv", index=False)
        print(f"✅ Predictions saved to predictions/predictions_{max_date}.csv")

    # ---- Display Section ----
    if print_df:
        if max_date:
            print(f"predicted on data on/up to {max_date}:")
            if len(df['date'].unique()) == 1:
                df.drop(columns=['date'], inplace=True)
        else:
            print(f"only backtest metrics:")
        
        # Format metrics for printing
        if 'holding_time_percentage' in df.columns:
            df['holding_time_percentage'] = df['holding_time_percentage'].apply(lambda x: f"{x:.2%}")
        if 'win_rate' in df.columns:
            df['win_rate'] = df['win_rate'].apply(lambda x: f"{x:.2%}")
        if 'cagr' in df.columns:
            df['cagr'] = df['cagr'].apply(lambda x: f"{x:.2%}")
        if 'sharpe' in df.columns:
            df['sharpe'] = df['sharpe'].apply(lambda x: f"{x:.3f}")

        df.sort_values(by=['proba','win_rate'],ascending=False, inplace=True)
        print(tabulate(df, headers='keys', tablefmt='fancy_grid'))

    return None



        


    
    

    






    # allowed_keys = list(next(iter(my_d.values())).keys()) + ['ticker', None]

    # if sort_by not in allowed_keys:
    #     print(f"ERROR: sort_by must be None or one of: {allowed_keys}")
    #     return None

    # df_stats = pd.DataFrame.from_dict(my_d, orient='index')

    # if pred_df is not None and not pred_df.empty:
        


    # if sort_by == 'ticker':
    #     df_stats = df_stats.sort_index(ascending=True)
    # elif sort_by is not None:
    #     df_stats = df_stats.sort_values(by=sort_by, ascending=False)

    # # ---- Save prediction DataFrame ----
    # if with_pred and save_csv and pred_df is not None and not pred_df.empty:
    #     max_date = pred_df.index.get_level_values("date").max()
    #     date_str = pd.to_datetime(max_date).date().isoformat()
    #     os.makedirs("predictions", exist_ok=True)

    #     # Copy and safely reset index without mutating original pred_df
    #     pred_df_safe = pred_df.copy()
    #     pred_df_safe.index.names = ['_ticker', '_date']
    #     pred_df_out = pred_df_safe.reset_index()
    #     pred_df_out = pred_df_out.rename(columns={'_ticker': 'ticker', '_date': 'date'})

    #     pred_df_out.to_csv(f"predictions/predictions_{date_str}.csv", index=False)
    #     print(f"✅ Predictions saved to predictions/predictions_{date_str}.csv")

    # # ---- Display Section ----
    # if print_df:
    #     if with_pred and pred_df is not None and not pred_df.empty:
    #         # Ensure correct index names
    #         if pred_df.index.names != ["ticker", "date"]:
    #             pred_df.index.set_names(["ticker", "date"], inplace=True)

    #         unique_dates = pred_df.index.get_level_values("date").unique()
    #         multiple_dates = len(unique_dates) > 1
    #         max_date = unique_dates.max()
    #         date_str = pd.to_datetime(max_date).date().isoformat()

    #         if multiple_dates:
    #             df_display = pred_df.copy()  # MultiIndex: [ticker, date]
    #         else:
    #             df_display = pred_df.xs(key=max_date, level="date")  # index: ticker only

    #         df_display = df_display.merge(df_stats, left_index=True, right_index=True, how="left")

    #         # Reorder columns
    #         pred_cols = ['signal', 'proba_long', 'sma30', 'atr', 'garch_vol']
    #         available_pred_cols = [col for col in pred_cols if col in df_display.columns]
    #         backtest_cols = [col for col in df_stats.columns if col in df_display.columns]

    #         # Insert separator
    #         df_display.insert(len(available_pred_cols), '← backtest metrics on the right', '')

    #         # Final column order
    #         df_display = df_display[available_pred_cols + ['backtest metrics →'] + backtest_cols]

    #         # Format metrics for printing
    #         if 'holding_time_percentage' in df_display.columns:
    #             df_display['holding_time_percentage'] = df_display['holding_time_percentage'].apply(lambda x: f"{x:.2%}")
    #         if 'win_rate' in df_display.columns:
    #             df_display['win_rate'] = df_display['win_rate'].apply(lambda x: f"{x:.2%}")
    #         if 'cagr' in df_display.columns:
    #             df_display['cagr'] = df_display['cagr'].apply(lambda x: f"{x:.2%}")
    #         if 'sharpe' in df_display.columns:
    #             df_display['sharpe'] = df_display['sharpe'].apply(lambda x: f"{x:.3f}")

    #         if multiple_dates:
    #             print(f"data as of multiple dates (prediction means whether to long the next trading day)")
    #         else:
    #             print(f"data as of {date_str} (prediction means whether to long the next trading day)")

    #         print(tabulate(df_display, headers='keys', tablefmt='fancy_grid'))

    #     else:
    #         df_display = df_stats.copy()
    #         df_display['holding_time_percentage'] = df_display['holding_time_percentage'].apply(lambda x: f"{x:.2%}")
    #         df_display['win_rate'] = df_display['win_rate'].apply(lambda x: f"{x:.2%}")
    #         df_display['cagr'] = df_display['cagr'].apply(lambda x: f"{x:.2%}")
    #         df_display['sharpe'] = df_display['sharpe'].apply(lambda x: f"{x:.3f}")

    #         print("data as of N/A (no prediction run)")
    #         print(tabulate(df_display, headers='keys', tablefmt='fancy_grid'))

    # return None
