import pandas_datareader as pdr
key="aef715b3799145152abfc85762ca74b1fd62277e"
df = pdr.get_data_tiingo('AAPL', api_key=key)
df.to_csv('AAPL.csv')