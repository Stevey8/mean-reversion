watchlist = [
    'nvda','msft','aapl','amzn','avgo',
    'meta','nflx','tsla','googl','cost',
    'goog','pltr','csco','amd','tmus',
    'lin','intu','pep','txn','isrg',
    'bkng','qcom','amgn','adbe','amat',
    'spot','tsm','rddt','mstr','rklb',
]

# watchlist = ['nvda','spot','pltr']

target_long_threshold = 0.02
target_long_lookahead = 5

drop_feats = [
    # 'date',
    'high','low','sma30','sma10','std30',
    'bollinger_upper','bollinger_lower',
    'rsi_smooth',
    'adx_pos', 'adx_neg'
]

backtest_config = {
    'proba_threshold': 0.6,
    'use_vol': 'garch_vol',
    'take_profit': 1.5,
    'stop_loss': 1,
}
