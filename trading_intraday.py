import alpaca_trade_api as tradeapi

key_path = '/Users/yochainusan/Desktop/backtest_intraday/config/alpaca/key.txt'
secret_path = '/Users/yochainusan/Desktop/backtest_intraday/config/alpaca/secret.txt'

# authentication and connection details
api_key = open(key_path, 'r').read()
api_secret = open(secret_path, 'r').read()
base_url = 'https://paper-api.alpaca.markets'

# instantiate REST API
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

# obtain account information
account = api.get_account()
print(account)