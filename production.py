from config import watchlist, drop_feats, param_grid, backtest_config
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

from datetime import datetime
from sklearn.model_selection import train_test_split

# -----

def run_pred(ticker, end_date):
    try: 
        df = get_data(ticker, end_date=end_date)
        if not df.empty:
            my_d = {
                'ohlcv': df.copy(),
                'data_as_of': end_date
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

    fitted_train = train_rscv(X_train, y_train, drop_feats, n_iter=30)
    my_d['train_set_model'] = {
        'model': fitted_train,
        # 'trained_at': datetime.today().date().isoformat(),
        'best_estimator': fitted_train.best_estimator_,
        'best_params': fitted_train.best_params_,
        'best_score': fitted_train.best_score_,
    }


    # pred and backtest

    df_trade, trade_stats = backtest(X_test, fitted_train, **backtest_config)
    total_trading_days = trade_stats['total_trading_days']

    my_d['test_set_result'] = {
        'backtest_config': backtest_config,
        **trade_stats,
        'cagr': get_cagr(df_trade),
        'sharpe': get_sharpe(df_trade, total_trading_days)
    }

    # train the final model on all data
    final_model = train_with_best_param(X, y, drop_feats, fitted_train)
    my_d['final_model'] = final_model
    my_d['last_update'] = datetime.today().date().isoformat()

    return my_d


# -----

def build_or_update_dfs(
        watchlist, 
        end_date = datetime.today().date().isoformat(),
        load_existing=True,
        force_retrain=False,
        tickers_to_retrain=None, 
):
    if load_existing:
        dfs = ...
    else:
        dfs = {}

    tickers_to_retrain = tickers_to_retrain or []


    for ticker in watchlist:
        if ticker in tickers_to_retrain: # should train
            print(f"{ticker} explicitly retraining")
            dfs[ticker] = run_pred(ticker, end_date)
        elif ticker in dfs and not force_retrain:
            print(f"{ticker} skipped (trained at {dfs[ticker]['last_update']})")
            continue
        else: # should train
            print(f"{ticker} training (new or forced)")
            dfs[ticker] = run_pred(ticker, end_date)

    return dfs

    


# print(f"most recent activity:") # also be mindful there are numerous tickers
# print(f"data as of: {}")
# print(f"model last trained at: {}")


