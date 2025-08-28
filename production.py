from typing import Iterable
from config import (
    watchlist, wrangling_config, backtest_config_1, backtest_config_2, backtest_config_3
)
from utils.dataloader import (
    save_pkl, read_pkl,
    get_data, get_sp500, get_metadata, 
    select_global_candidates, 
    feature_engineering, 
    market_trend, make_market_trend_dict, 
    feature_engineering_market, feature_engineering_metadata,
    create_target_long, 
    wrangling,
)
from utils.strategy import (
    XGB, 
    get_param_grid, get_drop_feats_indiv, get_drop_feats_global,
    get_ohe_feats, get_pipeline_indiv, get_pipeline_global,
    train_rscv, train_with_best_param, pred_proba_to_signal,
    get_prc_stats, plot_prc_with_thresholds, 
    entry_exit, backtest,
)

import os
import pytz
import pickle
# import logging # for future logging on cloud
import traceback
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from datetime import date, datetime, timedelta

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from tabulate import tabulate


# ======================================
# ===== some utility functions  =====
# ======================================


def get_available_date(now: datetime = None) -> datetime.date:
    """
    Return today's date if it's a weekday after 4:30 PM,
    otherwise return the most recent weekday before today.
    
    - If Saturday or Sunday: return Friday.
    - If Monday before 4:30 PM: return Friday.
    """
    if now is None:
        now = datetime.now()

    # weekday(): Monday=0, Sunday=6
    weekday = now.weekday()

    # If weekend -> roll back to Friday
    if weekday == 5:  # Saturday
        return (now - timedelta(days=1)).date()
    elif weekday == 6:  # Sunday
        return (now - timedelta(days=2)).date()

    # Weekday (Mon–Fri)
    cutoff = now.replace(hour=16, minute=30, second=0, microsecond=0)

    if now >= cutoff:
        # After 4:30 PM, return today
        return now.date()
    else:
        # Before 4:30 PM, go back one weekday
        if weekday == 0:  # Monday before cutoff
            return (now - timedelta(days=3)).date()  # last Friday
        else:
            return (now - timedelta(days=1)).date()

    

def split_4types(df_prepped,split_type):
    # make sure no NaN 
    df = df_prepped.dropna()
    X = df.drop(columns='target_long')
    y = df['target_long']

    # type can only be 1,2,3,4
    if split_type == 1: 
        # will be used in global: 
        # criteria: since < 2016-01-01 and is_sp500
        # if selected, train up to 2020-12-31 (guaranteed 5 years of train data)
        # val 2021-2022, test 2023 onwards
        X_train = X.copy()[:'2020-12-31']
        y_train = y.copy()[:'2020-12-31']
        X_val = X.copy()['2021-01-01':'2022-12-31']
        y_val = y.copy()['2021-01-01':'2022-12-31']
        X_test = X.copy()['2023-01-01':]
        y_test = y.copy()['2023-01-01':]
    elif split_type == 2:
        # not used in global while having sufficient data
        # criteria: since < 2018-01-01 (guaranteed 5 years of data)
        # train up to 2022-12-31, test 2023 onwards
        X_train = X.copy()[:'2022-12-31']
        y_train = y.copy()[:'2022-12-31']
        X_val = None
        y_val = None
        X_test = X.copy()['2023-01-01':]
        y_test = y.copy()['2023-01-01':]
    elif split_type == 3:
        # less data but has at least 5 years of data up to 2025-06-30
        # train exactly 5 years of data, test rest
        train_end = X.index.min() + pd.DateOffset(years=5)
        test_start = train_end + pd.DateOffset(days=1)
        X_train = X.copy()[:train_end]
        y_train = y.copy()[:train_end]
        X_val = None
        y_val = None
        X_test = X.copy()[test_start:]
        y_test = y.copy()[test_start:]
    elif split_type == 4: # type 4
        # no sufficient data, do not split
        # for prediction simply use the global model to predict
        X_train = None
        y_train = None
        X_val = None
        y_val = None
        X_test = None
        y_test = None
    else:
        raise ValueError("type must be 1, 2, 3, or 4")
    return {
        'X': X,
        'y': y,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }



# =====================
# ||                 ||
# ||   update data   ||
# ||                 ||
# =====================

def update_data_pkl(
        watchlist = watchlist, # note: sp500 tickers will be automatically added
        data = None,
        end_date = None,
        include_all_sp500_in_watchlist = False,
        reselect_global_candidates = False, # set to False to avoid reselection
        path = "files/data.pkl",
        global_candidates_path = "files/df_global_candidates.pkl",
        redo = False, # will update all data; otherwise only add missing data
        **kwargs, # for target creation i.e. wrangling_config
):
    wrangling_params = {**wrangling_config, **kwargs}

    if data is None:
        data = read_pkl(path, redo)  # will handle `not os.path.exists(path)` too
    if end_date is None:
        end_date = get_available_date()
    else:
        print(f"update data up to {end_date}")
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date() if isinstance(end_date, str) else end_date

    etfs = make_market_trend_dict()

    # get the df for selected global candidates 
    # this df contains the exact sp500 tickers
    df_global_candidates = select_global_candidates(
        path = global_candidates_path,
        redo = reselect_global_candidates
    )

    ls_sp500 = df_global_candidates['ticker'].tolist()
    ls_global_candidates = df_global_candidates.loc[
        df_global_candidates['selected'], 'ticker'
    ].tolist()

    # make sure everything is lowercase 
    ls_sp500 = [i.lower() for i in ls_sp500]
    ls_global_candidates = [i.lower() for i in ls_global_candidates]
    watchlist = [i.lower() for i in watchlist]

    if include_all_sp500_in_watchlist:
        watchlist = list(set(watchlist + ls_sp500))
    else: # only the selected ones (essential for global model)
        watchlist = list(set(watchlist + ls_global_candidates))
        
    print(f"total number of tickers to process: {len(watchlist)}")


    any_update = False
    # only update the data specifried by watchlist; keep existing
    with tqdm(watchlist, desc="Processing tickers", leave=True) as pbar:
        for i in pbar:
            # always show which ticker we're touching
            pbar.set_postfix_str(f"Last: {i}")

            # skip if already up-to-date
            if (
                i in data
                and data[i].get('as_of', None) is not None
                and data[i]['as_of'] >= end_date
            ):
                continue

            try:
                any_update = True
                # --- CREATE or UPDATE OHLCV ---
                if i not in data:
                    ohlcv = get_data(i, start_date='1999-01-01', end_date=end_date)
                    prepped = wrangling(
                        ohlcv,
                        etfs=etfs,
                        sp500_tickers=ls_sp500,
                        **wrangling_params
                    )

                    since = ohlcv.index.min().date()
                    sector = prepped['sector'].iloc[0]
                    use_in_global = i in ls_global_candidates
                
                    # split type and split data 
                    if use_in_global:
                        split_type = 1 # use in global model
                    elif since < datetime.strptime("2018-01-01", "%Y-%m-%d").date():
                        split_type = 2  # train up to 2022-12-31 and rest is test
                    elif since < datetime.strptime("2020-06-30", "%Y-%m-%d").date():
                        split_type = 3  # use exactly 5yrs of data as train and rest is test
                    else: 
                        split_type = 4 # no sufficient data, do not split

                    split = split_4types(prepped,split_type)

                    data[i] = {
                        'ohlcv': ohlcv,
                        'as_of': ohlcv.index.max().date(),
                        'prepped': prepped,
                        'target_kwargs': {**wrangling_params}, 
                        'since': since,
                        'sector': sector,
                        'use_in_global': use_in_global,
                        'split_type': split_type,
                        'split': split
                    }


                else:
                    # check if there is new data, if not then skip
                    as_of = data[i]['as_of']
                    start_date = as_of + timedelta(days=1)
                    if start_date > end_date:
                        continue
                    new_ohlcv = get_data(i, start_date=start_date, end_date=end_date)
                    if new_ohlcv.empty:
                        continue

                    ohlcv = pd.concat([data[i]['ohlcv'], new_ohlcv])
                    ohlcv = ohlcv[~ohlcv.index.duplicated(keep='last')].sort_index()

                    # although not efficient, recalculate everything for now to ensure consistency
                    # caveats if not recalculating:
                    # 1) GARCH needs full history (and this takes most of the processing time)
                    # 2) cannot get the correct year_since unless modify the feature_engineering function
                    prepped = wrangling(
                        ohlcv,
                        etfs=etfs,
                        sp500_tickers=ls_sp500,
                        **wrangling_params
                    )

                    data[i] |= {
                        'ohlcv': ohlcv,
                        'as_of': ohlcv.index.max().date(),
                        'prepped': prepped,
                    }

                    new_start_idx = new_ohlcv.index.min()
                    new_prepped = prepped[new_start_idx:].dropna()
                    X_new_prepped = new_prepped.drop(columns='target_long')
                    y_new_prepped = new_prepped['target_long']
                    X = pd.concat([data[i]['split']['X'], X_new_prepped])
                    y = pd.concat([data[i]['split']['y'], y_new_prepped])
                    X = X[~X.index.duplicated(keep='last')].sort_index()
                    y = y[~y.index.duplicated(keep='last')].sort_index()

                    data[i]['split'] |= {
                        'X': X,
                        'y': y,
                    }

                    if data[i]['split_type'] in (1,2,3): 
                        X_test = pd.concat([data[i]['split']['X_test'], X_new_prepped])
                        y_test = pd.concat([data[i]['split']['y_test'], y_new_prepped])
                        X_test = X_test[~X_test.index.duplicated(keep='last')].sort_index()
                        y_test = y_test[~y_test.index.duplicated(keep='last')].sort_index()

                        data[i]['split'] |= {
                            'X_test': X_test,
                            'y_test': y_test
                        }

                    # for type 1 and 2, if use_in_global changed due to reselection
                    # then should resplit train and val sets
                    if (
                        reselect_global_candidates 
                        and data[i]['split_type'] in (1,2)
                        and (i in ls_global_candidates) != data[i]['use_in_global']
                    ):
                        use_in_global = i in ls_global_candidates
                        data[i]['use_in_global'] = use_in_global
                        if use_in_global:
                            split_type = 1
                        else:
                            split_type = 2
                        split = split_4types(data[i]['prepped'],split_type)
                        data[i]['split_type'] = split_type

            except Exception as e:
                pbar.write(f"[ERROR] {i}: {e}")
                traceback.print_exc()
                if i in data:
                    del data[i]
                continue

    if any_update:
        print('data done. saving...')
        save_pkl(path, data)
    return data




# ===================================
# ||                               ||
# ||   train models and backtest   ||
# ||                               ||
# ===================================
def update_models_pkl(
        watchlist = watchlist, # indiv models trained based on this
        models = None,
        data = None,
        update_data = True, # new data comes in everyday
        update_global = False, # frozen
        update_indiv = False, # frozen (end date depends on split type)
        update_indiv_fullset = True, # new data comes in everyday
        update_stack = False, # frozen
        update_backtest = True,
        reselect_global_candidates = False,
        redo = False,
        models_path = 'files/models.pkl',
        data_path = 'files/data.pkl',
        metrics_path = 'files/metrics.pkl',
):
    if models is None:
        models = read_pkl(models_path, redo=redo)
    for i in ['global', 'indiv_trainset', 'indiv_fullset']:
        if i not in models:
            models[i] = {}
    if redo: 
        update_global = True
        update_indiv = True
        update_indiv_fullset = True
        update_stack = True
        update_backtest = True


    # whether to use up to latest data
    if data is None: 
        if update_data:
            print('updating data...')
            data = update_data_pkl(
                watchlist = watchlist,
                reselect_global_candidates=reselect_global_candidates
            )
        else: 
            print('gathering data without updating...')
            data = read_pkl(data_path, redo=False)

    # retain a copy of global tickers
    use_in_global_tickers = []
    for k,v in data.items():
        if v.get('use_in_global', False):
            use_in_global_tickers.append(k)

    watchlist = list(set(watchlist + use_in_global_tickers))

    if update_global: 
        X_train_global, y_train_global = [], []
        drop_feats_global = get_drop_feats_global()
        ohe_feats = get_ohe_feats()
        pipe_global = get_pipeline_global(drop_feats_global, ohe_feats)
        for k,v in data.items():
            if v.get('use_in_global', False):
                X_train_global.append(v['split']['X_train'])
                y_train_global.append(v['split']['y_train'])
        print(f"training global model with {len(use_in_global_tickers)} tickers...")
        X_train_global = pd.concat(X_train_global)
        y_train_global = pd.concat(y_train_global)

        rscv_global = train_rscv(X_train_global, y_train_global, pipe_global)
        models['global'] = {
            'model': rscv_global,
            'trained_on': datetime.now().date(), # call it trained_on instead of as_of
            'data_as_of': X_train_global.index.max().date(),
        }
        print("global model done. saving models...")
        save_pkl(models_path, models)
        print('saving global done')

    if update_indiv:
        drop_feats_indiv = get_drop_feats_indiv()
        pipe_indiv = get_pipeline_indiv(drop_feats_indiv)
        print(f"amount of indiv models: {len(watchlist)}")
        
        with tqdm(watchlist, desc="training indiv models", leave=True) as pbar:
            for t in pbar:
                if t not in data:
                    tqdm.write(f"[ERROR] {t} not in data.pkl")
                    continue
                if data[t]['split_type'] == 4:
                    tqdm.write(f"Skipping {t} due to insufficient data")
                    continue

                if (
                    t in models['indiv_trainset']
                    and models['indiv_trainset'][t]['data_as_of'] == data[t]['split']['X_train'].index.max().date()
                ): 
                    pbar.set_postfix_str(f"{t} has up-to-date indiv_trainset; checking fullset...")
                else: 
                    pbar.set_postfix_str(f"training {t} indiv_trainset...")
                    X_train = data[t]['split']['X_train']
                    y_train = data[t]['split']['y_train']
                    if any([
                    X_train is None,
                    X_train.empty,
                    y_train is None,
                    y_train.empty,
                    ]):
                        pbar.set_postfix_str(f"Skipping {t} due to missing data")
                        continue
                    pbar.set_postfix_str(f"Training {t}")
                    rscv_indiv = train_rscv(X_train, y_train, pipe_indiv)
                    models['indiv_trainset'][t] = {
                        'model': rscv_indiv,
                        'trained_on': datetime.now().date(),
                        'data_as_of': X_train.index.max().date(),
                    }
                    pbar.set_postfix_str(f"Finished training {t} indiv_trainset.")

                if update_indiv_fullset: 
                    # X_all = data[t]['prepped'].dropna().drop(columns='target_long')
                    # y_all = data[t]['prepped'].dropna()['target_long']
                    X_all = data[t]['split']['X']
                    y_all = data[t]['split']['y']
                    if y_all.isnull().any(): # won't run into type 4
                        raise ValueError(f"y_all for {t} contains NaN")
                    rscv_indiv = models['indiv_trainset'][t]['model']
                    rscv_all = train_with_best_param(X_all, y_all, pipe_indiv, rscv_indiv)
                    models['indiv_fullset'][t] = {
                        'model': rscv_all,
                        'trained_on': datetime.now().date(),
                        'data_as_of': X_all.index.max().date(), # only a single 'as_of' without 'data_as_of'
                    }

        print("indiv models done. saving models...")
        save_pkl(models_path, models)
        print('saving indiv done')

    if update_stack: 
        # make sure all indiv and global models are trained 
        # add a property as_of to each indiv so that can make sure not to retrain if not needed
        print("training stack model...")
        if 'global' not in models or 'indiv_trainset' not in models:
            raise ValueError("Please train indiv and global models before training stack model")
        if 'stack' not in models:
            models['stack'] = None
        global_model = models['global']['model']
        X_stack = []
        y_stack = []
        for t in use_in_global_tickers:
            if t not in models['indiv_trainset']:
                raise ValueError(f"Please train indiv model for {t} before training stack model")
            if any([
                data[t]['split']['X_val'] is None,
                data[t]['split']['X_val'].empty,
                data[t]['split']['y_val'] is None,
                data[t]['split']['y_val'].empty,
            ]):
                raise ValueError(f"Please make sure {t} has a validation set before training stack model")
            # if models['indiv_trainset'][t]['as_of'] != models['global']['as_of']:
            #     raise ValueError(f"Please retrain indiv model for {t} / global model to match before training stack model")
            indiv_model = models['indiv_trainset'][t]['model']

            X_val = data[t]['split']['X_val']
            y_val = data[t]['split']['y_val']

            y_proba_global = global_model.predict_proba(X_val)[:, 1]
            y_proba_indiv = indiv_model.predict_proba(X_val)[:, 1]
            df_proba_val = pd.DataFrame({
                'global_proba': y_proba_global,
                'indiv_proba': y_proba_indiv
            }, index=X_val.index)
            
            X_stack.append(df_proba_val)
            y_stack.append(y_val)
        X_stack = pd.concat(X_stack)
        y_stack = pd.concat(y_stack)
        lr = LogisticRegression()
        lr.fit(X_stack, y_stack)
        models['stack'] = {
            'model': lr,
            'trained_on': datetime.now().date(),
            'data_as_of': y_stack.index.max().date(),
        }
        print("stack model done")

    
    if update_backtest: 
        metrics = {
            'backtested': {},
            'skipped': [],
            'backtest_config': {
                'cat1': backtest_config_1,
                'cat2': backtest_config_2,
                'cat3': backtest_config_3,
            },
            'last_updated': datetime.now().date()
        }
        skip_backtest_tickers = []
        for t in tqdm(watchlist, desc="backtesting"):
            split_type = data[t]['split_type']
            if split_type == 4:
                tqdm.write(f"Skipping {t} due to insufficient data (type 4)") #TODO do we want to pred with global using the whole set?
                skip_backtest_tickers.append(t)
                continue


            X_test = data[t]['split']['X_test']
            y_test = data[t]['split']['y_test']

            if any([
                X_test is None,
                X_test.empty,
                y_test is None,
                y_test.empty,
            ]):
                tqdm.write(f"Skipping {t} due to missing test data")
                skip_backtest_tickers.append(t)
                continue

            if split_type == 1:
                backtest_config = backtest_config_1
            elif split_type == 2:
                backtest_config = backtest_config_2
            elif split_type == 3:
                backtest_config = backtest_config_3

            global_model = models['global']['model']
            y_proba_global = global_model.predict_proba(X_test)[:, 1]
            if t in models['indiv_trainset']:
                indiv_model = models['indiv_trainset'][t]['model']
                y_proba_indiv = indiv_model.predict_proba(X_test)[:, 1]
                df_proba_test = pd.DataFrame({
                    'global_proba': y_proba_global,
                    'indiv_proba': y_proba_indiv
                }, index=X_test.index)
                backtest_model = models['stack']['model']
            else: 
                df_proba_test = pd.DataFrame({
                    'global_proba': y_proba_global,
                }, index=X_test.index)
                backtest_model = models['global']['model']
            
            _, trade_stats = backtest(
                X_backtest_input = X_test,
                model = backtest_model,
                **backtest_config,
                X_model_input=df_proba_test
            )

            threshold = backtest_config['proba_threshold']
            y_final_proba = backtest_model.predict_proba(df_proba_test)[:, 1]
            y_final_pred = pred_proba_to_signal(y_final_proba, threshold)

            precision = precision_score(y_test, y_final_pred)
            recall = recall_score(y_test, y_final_pred)
            prevalence = y_test.mean()

            metrics['backtested'][t] = {
                'precision': precision,
                'recall': recall,
                'prevalence': prevalence,
                **trade_stats,
                'cat': split_type,
            }

        metrics['skipped'] = skip_backtest_tickers
        save_pkl(metrics_path, metrics)

    save_pkl(models_path, models)
    return models



# ========================
# ||                    ||
# ||   check backtest   ||
# ||                    ||
# ========================

def check_backtest(tickers: Iterable[str] = None, metrics_path = 'files/metrics.pkl'):
    if os.path.exists(metrics_path):
        metrics = read_pkl(metrics_path, redo=False)
        if tickers is None: # or not none
            ...
        if metrics:
            df_metrics = pd.DataFrame(metrics['backtested']).T
            df_metrics = df_metrics.rename(columns={
                'total_trading_days': 'd_trade',
                'total_holding_days': 'd_hold',
                'holding_time_percentage': 'd_%',
                'expectancy': 'exp'
            })

            if tickers is not None:
                # accept a single string too
                if isinstance(tickers, str):
                    tickers = [tickers]
                # keep only those present; preserves the input order
                present = [t for t in tickers if t in df_metrics.index]
                df_metrics = df_metrics.loc[present]

            print(tabulate(df_metrics, headers='keys', tablefmt='psql', floatfmt=".4f"))
            print("cat legend:\n"
                  "1 = used in global and stack; stacked results from indiv (trained with 5+ years) & global (in-sample); complete test set starting 2023-01-01\n"
                  "2 = not used in global; stacked results from indiv (trained with 5+ years) & global (out-of-sample); complete test set starting 2023-01-01\n"
                  "3 = not used in global; stacked results from indiv (trained with 5 years) & global (out-of-sample); shortened test set\n"
            )
            print(f"cat 4 aka skipped tickers due to insufficient data: {metrics.get('skipped', [])}")
            print(f"backtest config: {metrics.get('backtest_config', {})}")
        else:
            print("No metrics found in metrics.pkl")
    else:
        print("metrics.pkl not found. Please run update_models_pkl with update_backtest=True first.")

            

# ============================================
# ||                                        ||
# ||   run pred and show backtest results   ||
# ||                                        ||
# ============================================

def run_pred(
        watchlist = watchlist,
        data = None, 
        models = None, 
        metrics = None,
        show_all = False,
        cutoff = 0.5,
        # date = None, # right now only run the last available row of data only
        models_path = "files/models.pkl",
        metrics_path = "files/metrics.pkl"
):
    if data is None: 
        data = update_data_pkl() # always update data 
    if models is None: 
        models = read_pkl(models_path, redo=False)
    if metrics is None: 
        metrics = read_pkl(metrics_path, redo=False)

    ls_data_not_available = []
    d_pred_date = {}
    results = []
    for t in watchlist:
        if t not in data:
            ls_data_not_available.append(t)
            continue
        to_pred = data[t]['prepped'].tail(1).drop(columns='target_long')
        pred_date = to_pred.index[0]
        if pred_date not in d_pred_date:
            d_pred_date[pred_date] = [t]
        d_pred_date[pred_date].append(t)

        if data[t]['split_type'] == 4:
            model = models['global']['model']
            proba_gonly = model.predict_proba(to_pred)[:, 1]
            results.append({
                'ticker': t,
                'proba_sfull': None,
                'proba_strain': None,
                'proba_g': proba_gonly[0],
                'proba_ifull': None,
                'proba_itrain': None,
                'cat': 4,
            })
        else:
            model_global = models['global']['model']
            model_indiv_full = models['indiv_fullset'][t]['model']
            model_indiv_train = models['indiv_trainset'][t]['model']
            proba_g = model_global.predict_proba(to_pred)[:, 1]
            proba_ifull = model_indiv_full.predict_proba(to_pred)[:, 1]
            proba_itrain = model_indiv_train.predict_proba(to_pred)[:, 1]
            stack = models['stack']['model']
            proba_sfull = stack.predict_proba(pd.DataFrame({
                'global_proba': proba_g,
                'indiv_proba': proba_ifull
            }, index=to_pred.index))[:, 1]
            proba_strain = stack.predict_proba(pd.DataFrame({
                'global_proba': proba_g,
                'indiv_proba': proba_itrain
            }, index=to_pred.index))[:, 1]
            results.append({
                'ticker': t,
                'proba_sfull': proba_sfull[0],
                'proba_strain': proba_strain[0],
                'proba_g': proba_g[0],
                'proba_ifull': proba_ifull[0],
                'proba_itrain': proba_itrain[0],
                'cat': data[t]['split_type'],
            })
    df_results = pd.DataFrame(results).set_index('ticker').sort_values(by='proba_sfull', ascending=False)

    if ls_data_not_available:
        print(f"Data not available for the following tickers: {', '.join(ls_data_not_available)}")
    if len(d_pred_date) == 1:
        print(f"prediction of data on {list(d_pred_date.keys())[0]}")
    else: 
        print('inconsistent data dates:')
        print(d_pred_date)

    if show_all:
        print(tabulate(df_results, headers='keys', tablefmt='psql', floatfmt=".4f"))
        print("please check individual backtest result by calling check_backtest([tickers])")
    else: 
        proba_cols = ["proba_sfull", "proba_strain", "proba_g", "proba_ifull", "proba_itrain"]
        df_filtered = df_results[(df_results[proba_cols] >= cutoff).all(axis=1)]
        print(tabulate(df_filtered, headers='keys', tablefmt='psql', floatfmt=".4f"))
        tickers = df_filtered.index.tolist()
        print("\n" + "="*20 + " BACKTEST RESULTS " + "="*20)
        print("╔" + "═"*58 + "╗")
        print("║{:^58s}║".format("BACKTEST SUMMARY"))
        print("╚" + "═"*58 + "╝\n")
        check_backtest(tickers)
