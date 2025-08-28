import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import timedelta, datetime, date

from sklearn.model_selection import RandomizedSearchCV, train_test_split, TimeSeriesSplit
from sklearn.base import BaseEstimator, TransformerMixin, clone, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    make_scorer, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, 
    roc_curve, RocCurveDisplay, auc, roc_auc_score, precision_recall_curve, PrecisionRecallDisplay, 
    precision_score, recall_score, f1_score, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier, plot_importance



# =======================
# ===== XGB Wrapper =====
# =======================

class XGB(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.xgb_params = kwargs
        self.model = None

    def fit(self, X, y):
        y = np.asarray(y)
        pos = np.sum(y == 1)
        neg = np.sum(y == 0)
        scale_pos_weight = neg / pos if pos != 0 else 1.0

        self.model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            objective='binary:logistic',
            **self.xgb_params
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        return self.xgb_params

    def set_params(self, **params):
        self.xgb_params.update(params)
        return self
    


# =======================
# ===== Train Model =====
# =======================

def get_param_grid():
    return {
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__n_estimators': [100, 300, 500],
        'xgb__subsample': [0.5, 0.7, 1.0],
        'xgb__colsample_bytree': [0.5, 0.7, 1.0]
    }

def get_drop_feats_indiv():
    drop_feats = [
        'high','low','sma30','sma10','std30',
        'bollinger_upper','bollinger_lower',
        'rsi_smooth',
        'adx_pos', 'adx_neg',
        'ticker',
        'sector', 'market_cap', 'avg_volume', 'beta', 'is_sp500'
    ] 
    return drop_feats

def get_drop_feats_global():
    drop_feats = [
        'high','low','sma30','sma10','std30',
        'bollinger_upper','bollinger_lower',
        'rsi_smooth',
        'adx_pos', 'adx_neg',
        'is_sp500',
        'ticker', # drop ticker too
    ]
    return drop_feats


def get_ohe_feats():
    ohe_feats = [
        'sector',
    ]
    return ohe_feats

def get_pipeline_indiv(drop_feats):
    ct = make_column_transformer(
        ('drop', drop_feats),  
        remainder='passthrough'  
    )
    pipe = Pipeline([
        ('ct', ct),
        ('xgb', XGB(random_state=42))
    ])
    return pipe

def get_pipeline_global(drop_feats, ohe_feats):
    ct = make_column_transformer(
        ('drop', drop_feats),
        (OneHotEncoder(handle_unknown='ignore', sparse_output=False), ohe_feats),
        remainder='passthrough'  
    )
    pipe = Pipeline([
        ('ct', ct),
        ('xgb', XGB(random_state=42))
    ])
    return pipe



def train_rscv(X_train, y_train, pipe, n_iter=30):
    rscv = RandomizedSearchCV(
        pipe,
        param_distributions=get_param_grid(),
        n_iter=n_iter,
        scoring='f1',
        cv=3,
        verbose=0,
        n_jobs=-1,
        random_state=42
    )
    rscv.fit(X_train, y_train)
    return rscv



def train_with_best_param(X, y, pipe, fitted_rscv):
    best_params = {k: v for k, v in fitted_rscv.best_params_.items() if k.startswith('xgb__')}
    best_model = pipe.set_params(**best_params)
    best_model.fit(X, y)
    return best_model


# ===========================
# ===== Pred & Backtest =====
# ===========================

def get_prc_stats(y_test, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    return precision, recall, thresholds, avg_precision

def plot_prc_with_thresholds(precision, recall, thresholds, avg_precision, threshold_markers=[0, 0.5, 0.6, 0.7], title='Precision-Recall Curve'):
    """
    Plots a Precision-Recall curve and marks specific threshold points.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR Curve (AP = {avg_precision:.4f})", lw=2)

    for t_val in threshold_markers:
        if t_val <= np.max(thresholds) and t_val >= np.min(thresholds):
            idx = np.argmin(np.abs(thresholds - t_val))
            plt.plot(recall[idx], precision[idx], 'o', label=f'Threshold = {t_val}', markersize=8)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]  # Reverse legend order
    plt.legend(handles, labels, title="Legend")
    plt.grid(True)
    plt.show()


def pred_proba_to_signal(y_proba, threshold=0.5):
    """
    Convert predicted probabilities to binary signals.
    1 if probability >= threshold, else 0.
    returns a Series of signals.
    """
    return (y_proba >= threshold).astype(int)


def entry_exit(df, use_vol=None, take_profit=1, stop_loss=1, min_return = 0.01): 
    '''
    FOR EACH INSTRUMENT (don't use the aggregated one that contains multiple tickers)
    takes in the X_test dataframe containing model signals,
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
        multiplier = df['atr']/ df['open']
    elif use_vol == 'garch_vol':
        vol_cap = df['garch_vol'].quantile(0.9)
        multiplier = (df['garch_vol'].clip(upper=vol_cap)) / 100
    else:
        multiplier = pd.Series(0.01, index=df.index)
    effective_profit_threshold = (take_profit * multiplier).clip(lower=min_return)

    while i < n - 6:  # we need at least 5 days ahead, plus trading at open tomorrow
        if df['model_signal'].iloc[i] == 1:
            entry_date = df['date'].iloc[i+1]
            entry_price = df['open'].iloc[i+1]
            exit_price = None
            exit_date = None
            holding = None
            exit_reason = None

            # exit logic:
            # since yfinance only provides daily data (open, high, low, close),
            # for simplicity and scrutiny (i.e. not realistic to always exit at high price),
            # we will only check the open price for the next 5 days (t+1 to t+5)
            for j in range(1, 6):  # check up to 5 days ahead
                if i + 1 + j >= n:
                    break

                next_price = df['open'].iloc[i + 1 + j] 
                ret = (next_price - entry_price) / entry_price

                # Exit Conditions
                if ret >= effective_profit_threshold.iloc[i]:  # profit target
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


def get_cagr(df_trades):
    if len(df_trades) > 0:
        start_date = df_trades['entry_date'].min()
        end_date = df_trades['exit_date'].max()
        n_years = (end_date - start_date).days / 365.25
        capital = 1
        for r in df_trades['return']:
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

def get_expectancy(df_trades):
    if len(df_trades) == 0:
        return 0

    wins = df_trades[df_trades['return'] > 0]
    losses = df_trades[df_trades['return'] <= 0]

    win_rate = len(wins) / len(df_trades)
    loss_rate = 1 - win_rate

    avg_win = wins['return'].mean() if not wins.empty else 0
    avg_loss = abs(losses['return'].mean()) if not losses.empty else 0

    expectancy = win_rate * avg_win - loss_rate * avg_loss
    return expectancy

def backtest(
        X_backtest_input, 
        model, 
        proba_threshold = 0.5, 
        use_vol=None, 
        take_profit=1, 
        stop_loss=1, 
        min_return = 0.01,
        X_model_input=None
):
    if X_model_input is None:
        X_model_input = X_backtest_input


    y_proba = model.predict_proba(X_model_input)[:, 1]
    max_proba = y_proba.max()
    signals = pred_proba_to_signal(y_proba, threshold=proba_threshold)
    X_test_signal = X_backtest_input.copy()
    X_test_signal = X_test_signal.assign(model_signal=signals, y_proba=y_proba)

    df_trade = entry_exit(X_test_signal, use_vol, take_profit, stop_loss, min_return)
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

        last_5 = ''
        last_5_returns = df_trade['return'].tail(5).tolist()
        for r in last_5_returns:
            if r > 0:
                last_5 += '+'
            else:
                last_5 += '-'
            

        cagr = get_cagr(df_trade)
        sharpe = get_sharpe(df_trade, total_holding_days)
        expectancy = get_expectancy(df_trade)

        trade_stats = {
            'max_proba': max_proba,
            'exit_reason_spread': df_trade['exit_reason'].value_counts(),
            'holding_days_spread': df_trade['holding_days'].value_counts(),
            'total_trading_days': int(total_trading_days),
            'total_holding_days': int(total_holding_days),
            'holding_time_percentage': holding_time_percentage,
            'n_trades': n_trades,
            'n_wins': n_wins,
            'win_rate': win_rate,
            'last_5': last_5,
            'cagr': cagr,
            'sharpe': sharpe,
            'expectancy': expectancy,
        }
    
    else:
        trade_stats = {
            'max_proba': max_proba,
            'exit_reason_spread': 'N/A',
            'holding_days_spread': 'N/A',
            'total_trading_days': 0,
            'total_holding_days': 0,
            'holding_time_percentage': 0,
            'n_trades': 0,
            'n_wins': 0,
            'win_rate': 0,
            'last_5': '',
            'cagr': 0,
            'sharpe': 0,
            'expectancy': 0,
        }

    return df_trade, trade_stats