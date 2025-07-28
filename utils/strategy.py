import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RandomizedSearchCV, train_test_split, TimeSeriesSplit
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    make_scorer, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, 
    roc_curve, RocCurveDisplay, auc, roc_auc_score, precision_recall_curve, PrecisionRecallDisplay, precision_score, recall_score, 
)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier, plot_importance


# TRAIN 

def get_pipeline(drop_feats, scale_pos_weight):
    ct = make_column_transformer(
        ('drop', drop_feats),  
        remainder='passthrough'  
    )
    pipe = Pipeline([
        ('ct', ct),
        ('xgb', XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=0
        ))
    ])
    return pipe


def get_param_grid():
    return {
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__n_estimators': [100, 300, 500],
        'xgb__subsample': [0.5, 0.7, 1.0],
        'xgb__colsample_bytree': [0.5, 0.7, 1.0]
    }



def train_rscv(X_train, y_train, drop_feats, n_iter=30):
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) # neg class / pos class
    pipe = get_pipeline(drop_feats, scale_pos_weight)
    rscv = RandomizedSearchCV(
        pipe,
        param_distributions=get_param_grid(),
        n_iter=n_iter,
        scoring='f1',
        cv=5,
        verbose=0,
        n_jobs=-1,
        random_state=42
    )
    rscv.fit(X_train, y_train)
    return rscv



def train_with_best_param(X, y, drop_feats, fitted_rscv):
    scale_pos_weight = len(y[y == 0]) / len(y[y == 1])
    pipe = get_pipeline(drop_feats, scale_pos_weight)
    best_params = {k: v for k, v in fitted_rscv.best_params_.items() if k.startswith('xgb__')}
    best_model = pipe.set_params(**best_params)
    best_model.fit(X, y)
    return best_model





# -----

# PRED AND BACKTEST

def pred_proba_to_signal(y_proba, threshold=0.5):
    """
    Convert predicted probabilities to binary signals.
    1 if probability >= threshold, else 0.
    returns a Series of signals.
    """
    return (y_proba >= threshold).astype(int)


def entry_exit(df, use_vol=None, take_profit=1, stop_loss=1): 
    '''
    FOR EACH INSTRUMENT (don't use the aggregated one that contains multiple tickers)
    takes in the X_test or ohlcv dataframe containing model signals,
    returns a df that contains entry and exit dates, prices, returns, and holding days
    use_vol: either 'atr' or 'garch_vol'
    take_profit and stop loss as percentage
    '''
    # if date as index
    if pd.api.types.is_datetime64_any_dtype(df.index): 
        df = df.reset_index().rename(columns={'Date': 'date'})

    trades = []
    i = 0
    n = len(df)
    if use_vol not in [None, 'atr', 'garch_vol']:
        raise ValueError("use_vol must be one of: None, 'atr' or 'garch_vol'")
    
    if use_vol == 'atr':
        multiplier = df['atr']
    elif use_vol == 'garch_vol':
        multiplier = df['garch_vol']
    else:
        multiplier = pd.Series(1, index=df.index)

    while i < n - 6:  # we need at least 5 days ahead, plus trading at open tomorrow
        if df['model_signal'].iloc[i] == 1:
            entry_date = df['date'].iloc[i+1]
            entry_price = df['open'].iloc[i+1]
            exit_price = None
            exit_date = None
            holding = None
            exit_reason = None

            for j in range(1, 6):  # check up to 5 days ahead
                if i + 1 + j >= n:
                    break

                next_price = df['open'].iloc[i + 1 + j]
                ret = (next_price - entry_price) / entry_price

                # Exit Conditions
                if ret >= take_profit * multiplier.iloc[i]:  # profit target
                    exit_price = next_price
                    exit_date = df['date'].iloc[i + 1 + j]
                    holding = j
                    exit_reason = 'profit_target'
                    break
                elif ret <= -stop_loss * multiplier.iloc[i]:  # stop loss
                    exit_price = next_price
                    exit_date = df['date'].iloc[i + 1 + j]
                    holding = j
                    exit_reason = 'stop_loss'
                    break
                elif df['open'].iloc[i + 1 + j] >= df['sma30'].iloc[i + 1 + j]:  # revert to SMA30
                    exit_price = next_price
                    exit_date = df['date'].iloc[i + 1 + j]
                    holding = j
                    exit_reason = 'revert_to_sma30'
                    break

            if exit_price is None:
                # Max holding (5th day)
                exit_price = df['open'].iloc[i + 1 + 5]
                exit_date = df['date'].iloc[i + 1 + 5]
                exit_reason = 'max_holding_expired'
                holding = 5

            trade_return = (exit_price - entry_price) / entry_price
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return': trade_return,
                'holding_days': holding,
                'exit_reason': exit_reason
            })

            i = i + holding  # skip to the day after exit
        else:
            i += 1

    if len(trades) > 0:
        return pd.DataFrame(trades)
    
    return pd.DataFrame(columns=[
        'entry_date',
        'exit_date',
        'entry_price',
        'exit_price',
        'return',
        'holding_days',
        'exit_reason'
    ])


def backtest(X_test, model, proba_threshold = 0.5, use_vol=None, take_profit=1, stop_loss=1):
    trade_stats = {} 

    y_proba = model.predict_proba(X_test)[:, 1]
    trade_stats['max_proba'] = y_proba.max()

    signals = pred_proba_to_signal(y_proba, threshold=proba_threshold)
    X_test_signal = X_test.assign(model_signal=signals, y_proba=y_proba)

    df_trade = entry_exit(X_test_signal, use_vol, take_profit, stop_loss)
    if len(df_trade) > 0:
        total_holding_days = df_trade['holding_days'].sum()
        total_trading_days = len(X_test_signal)
        if total_trading_days > 0:
            holding_time_percentage = total_holding_days/total_trading_days
        else: 
            holding_time_percentage = 0

        n_trades = len(df_trade)
        n_wins = len(df_trade[df_trade['return'] > 0])
        if n_trades > 0:
            win_rate = n_wins/n_trades
        else: 
            win_rate = 0

        trade_stats = {
            'exit_reason_spread': df_trade['exit_reason'].value_counts(),
            'holding_days_spread': df_trade['holding_days'].value_counts(),
            'total_trading_days': total_trading_days,
            'total_holding_days': total_holding_days,
            'holding_time_percentage': holding_time_percentage,
            'n_trades': n_trades,
            'n_wins': n_wins,
            'win_rate': win_rate
        }
    
    else:
        trade_stats = {
            'exit_reason_spread': 'not available',
            'holding_days_spread': 'not available',
            'total_trading_days': 0,
            'total_holding_days': 0,
            'holding_time_percentage': 0,
            'n_trades': 0,
            'n_wins': 0,
            'win_rate': 0
        }

    return df_trade, trade_stats


def get_cagr(df):
    if len(df) > 0:
        start_date = df['entry_date'].min()
        end_date = df['exit_date'].max()
        n_years = (end_date - start_date).days / 365.25
        capital = 1
        for r in df['return']:
            capital *= (1 + r)
        cagr = capital ** (1 / n_years) - 1
        return cagr
    return 0

def get_sharpe(df_trades, total_trading_days):
    if len(df_trades) > 0 and total_trading_days > 0:
        ret_mean = df_trades['return'].mean()
        ret_std = df_trades['return'].std()
        n_trades = len(df_trades)
        trades_per_year = (n_trades/total_trading_days) * 252
        return (ret_mean/ret_std) * np.sqrt(trades_per_year)
    return 0


# -----


# MODEL EVAL FOR TRAINING 

def plot_model_metrics(y_true, y_proba):
    y_pred = (y_proba >= 0.5).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[0], colorbar=False)

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    roc_disp.plot(ax=axes[1])

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    pr_disp.plot(ax=axes[2])

    plt.tight_layout()
    plt.show()



def eval_pipe(X, y, pipe, n_split=5, predict_proba_threshold=0.5):
    '''
    for evaluations on train set, make sure X and y does not contain anything from the test set (future data), 
    and use n_split=5 for 5-fold time series cross-validation
    '''
    tscv = TimeSeriesSplit(n_splits=n_split)
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= predict_proba_threshold).astype(int)

        print(f'Fold {i+1}:\n{classification_report(y_test, y_pred)}')

        if i == 0:
            print('Fold 1 visualized below')
            plot_model_metrics(y_test, y_proba)



# -----

# CHECK PRED RESULT
 
def check_return_stats(df):
    if len(df) > 0:
        print(f"check return stats: \n {df['return'].describe()}")
        sns.histplot(df['return'])
    print('no trade available')


def safe_format(value, fmt, default="N/A"):
    try:
        return format(value, fmt) if value is not None else default
    except (TypeError, ValueError):
        return default

def print_test_set_result(ticker, dfs):
    result = dfs[ticker]['test_set_result']
    
    print(
        f"ticker: {ticker} \n"
        f"total_holding_days: {result.get('total_holding_days', 'N/A')} \n"
        f"total_trading_days: {result.get('total_trading_days', 'N/A')} \n"
        f"holding_time_percentage: {safe_format(result.get('holding_time_percentage'), '.2%')} \n" 
        f"n_trades: {result.get('n_trades', 'N/A')} \n" 
        f"win_rate: {safe_format(result.get('win_rate'), '.2%')} \n" 
        f"cagr: {safe_format(result.get('cagr'), '.2%')} \n" 
        f"sharpe: {safe_format(result.get('sharpe'), '.3')} \n" 
        "---"
    )
