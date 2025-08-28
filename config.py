# first 25 is the top 26 from qqq (excluding goog because googl is presented) as of 2025-07-24

# watchlist = [
#     'nvda','msft','aapl','amzn','avgo',
#     'meta','nflx','tsla','googl','cost',
#     'pltr','csco','amd','tmus','lin',
#     'intu','pep','txn','isrg','bkng',
#     'qcom','amgn','adbe','amat','shop',
#     # above from qqq top 26
#     'ibm','orcl','snow','uber','xom',
#     'cvx','psx','jnj','pfe','lly',
#     'jpm','bac','gs','wmt','ko',
#     'dis','cat','ba','de','spot',
#     'v','ma','axp','cmg','anf',
#     # above from nyse
#     # more stocks below
#     'abnb','tsm','rddt','mstr','rklb',
#     'hon','crwd','snps','sbux','intc',
#     'mar','mrvl','pypl','adsk','asml',
#     'now','sofi','smci','dash','crm',
#     'ms','coin','roku','net','tmo',
#     'enph','fdx','regn','panw','uhn',
#     'nee','pld','apd','mo','vz', 
#     'duk','so','d','are','o',
#     'vtr','ecl','shw','nue','hsy',
#     'kmb','chd','t','cci','pwr',
# ]

# watchlist = [
#     'nvda','msft','aapl','meta','nflx',
#     'tsla','googl','cost','amzn','amd',
#     # 'pltr','spot','v','ma','axp',
#     # 'aem','xom','dvn','cvx','tsm',
#     # 'uber','team','snps','intc','smci',
#     # 'sbux','dis','shop','tmus','qcom',
#     # 'snow','now','sofi','coin','mstr',
#     # 'anf','ibm','orcl','txn','amat',
#     # 'adbe','adsk','asml','pypl','crm',
#     # 'crwd','crwv','jpm','rklb','rddt',
# ]

watchlist = [
    'nvda','msft','aapl','meta','nflx',
    'tsla','googl','cost','amzn','amd',
    'pltr','spot','v','ma','axp',
    'aem','xom','dvn','cvx','tsm',
    'uber','team','snps','intc','smci',
    'sbux','dis','shop','tmus','qcom',
    'snow','now','sofi','coin','mstr',
    'anf','ibm','orcl','txn','amat',
    'adbe','adsk','asml','pypl','crm',
    'crwd','crwv','jpm','rklb','rddt',
]


# watchlist = ['aapl']
# watchlist = ['msft']
# watchlist = ['nvda','pltr','rddt','anf','coin']


wrangling_config = {
    'timing': 'high', # for time t, check high price from t+1 to t+5
    'strategy': 'dynamic',
    'vol': 'garch_vol',
    'take_profit': 1.5,
    'min_return': 0.02,
}

# 3 backtest configs for 3 different categories (see check_backtest() in production.py for details)
# note: category 4 does not have enough data to run backtests

backtest_config_1 = {
    'proba_threshold': 0.6,
    'use_vol': 'garch_vol',
    'take_profit': 1.5,
    'stop_loss': 1,
    'min_return': 0.02,
}

backtest_config_2 = {
    'proba_threshold': 0.6,
    'use_vol': 'garch_vol',
    'take_profit': 1.5,
    'stop_loss': 1,
    'min_return': 0.02,
}

backtest_config_3 = {
    'proba_threshold': 0.6,
    'use_vol': 'garch_vol',
    'take_profit': 1.5,
    'stop_loss': 1,
    'min_return': 0.02,
}