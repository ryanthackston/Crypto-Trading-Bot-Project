
import bitfinex
import datetime
import time
import pandas as pd

# Create api instance of the v2 API
api_v2 = bitfinex.bitfinex_v2.api_v2()

# Define query parameters
pair = 'BTCUSD' # Currency pair of interest
TIMEFRAME = '1m'#,'4h','1h','15m','1m'

# Define the start date
t_start = datetime.datetime(2021, 11, 27, 0, 1)
t_start = time.mktime(t_start.timetuple()) * 1000

# Define the end date
t_stop = datetime.datetime(2021, 11, 30, 16, 0)
t_stop = time.mktime(t_stop.timetuple()) * 1000

# Download OHCL data from AP
result = api_v2.candles(symbol=pair, interval=TIMEFRAME, limit=10000, start=t_start, end=t_stop)

# Convert list of data to pandas dataframe
names = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
df = pd.DataFrame(result, columns=names)
df['Date'] = pd.to_datetime(df['Date'], unit='ms')
df2 = df.iloc[::-1]
# we can plot our downloaded data
import matplotlib.pyplot as plt
plt.plot(df2['Open'])
plt.show()


df2.to_csv('Test-Bitcoin-Data-Nov27-Nov30.csv', index=False)