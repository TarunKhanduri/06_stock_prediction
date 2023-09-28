import pandas_datareader as pdr
key=""
df = pdr.get_data_tiingo('AAPL', api_key=key)
df.to_csv('AAPL.csv')
