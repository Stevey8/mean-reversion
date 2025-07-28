# first 25 is the top 26 from qqq (excluding goog because googl is presented) as of 2025-07-24

watchlist = [
    'nvda','msft','aapl','amzn','avgo',
    'meta','nflx','tsla','googl','cost',
    'pltr','csco','amd','tmus','lin',
    'intu','pep','txn','isrg','bkng',
    'qcom','amgn','adbe','amat','shop',
    # above from qqq top 26
    'ibm','orcl','snow','uber','xom',
    'cvx','psx','jnj','pfe','lly',
    'jpm','bac','gs','wmt','ko',
    'dis','cat','ba','de','spot',
    'v','ma','axp','cmg','anf',
    # above from nyse
    # more stocks below
    'abnb','tsm','rddt','mstr','rklb',
    'hon','crwd','snps','sbux','intc',
    'mar','mrvl','pypl','adsk','asml',
    'now','sofi','smci','dash','crm',
    'ms','coin','roku','net','tmo',
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
