import yfinance as yf
import pandas as pd

# Set the date range for downloading historical data
start_date = '2008-01-01'
end_date = '2024-08-01'

# Download the historical data for the GLD ETF (which tracks gold prices)
# Auto-adjust is used to adjust the data for dividends and stock splits
data = yf.download('GLD', start_date, end_date, auto_adjust=True)

# Drop any rows with missing values to ensure clean data
data = data.dropna()

# Initialize a list to keep track of the names of the generated features
feature_list = []

# MOMENTUM: Calculate the 3-day Moving Average of the closing price
feature_name = 'MA3'
data[feature_name] = data['Close'].rolling(window=3).mean()
feature_list.append(feature_name)

# MOMENTUM: Calculate the 9-day Moving Average of the closing price
feature_name = 'MA9'
data[feature_name] = data['Close'].rolling(window=9).mean()
feature_list.append(feature_name)

# MOMENTUM: Calculate Moving Averages over different time windows (10, 15, 20, 25 days)
for i in range(10, 30, 5):
    feature_name = 'MA' + str(i)
    data[feature_name] = data['Close'].rolling(window=i).mean()
    feature_list.append(feature_name)

# MOMENTUM: Calculate Exponential Moving Averages (EMA) over different spans (8, 16, 24 days)
for i in range(8, 30, 8):
    feature_name = 'EMA_' + str(i)
    data[feature_name] = data['Close'].ewm(span=i, min_periods=i).mean()
    feature_list.append(feature_name)

# VOLATILITY: Calculate the rolling standard deviation (volatility) over different time windows (5, 10, 15 days)
for i in range(5, 20, 5):
    feature_name = 'STD_' + str(i)
    data[feature_name] = data['Close'].rolling(window=i).std()
    feature_list.append(feature_name)

# RETURNS: Calculate the sum of daily percentage changes over different time windows (3, 6, 9 days)
for i in range(3, 12, 3):
    feature_name = 'PCT_' + str(i)
    data[feature_name] = data['Close'].pct_change().rolling(i).sum()
    feature_list.append(feature_name)

# VOLUME: Calculate the 4-day Moving Average of the trading volume
feature_name = 'VM_4'
data[feature_name] = data['Volume'].rolling(4).mean()
feature_list.append(feature_name)

# DAILY MOMENTUM: Calculate the difference between the daily close and open prices (Candlestick Feature)
feature_name = 'CO'
data[feature_name] = data['Close'] - data['Open']
feature_list.append(feature_name)

# DAILY MOMENTUM: Calculate the daily range between the high and low prices (Candlestick Feature)
feature_name = 'HL'
data[feature_name] = data['High'] - data['Low']
feature_list.append(feature_name)

# Download additional data for related financial instruments and indices to use as features

# GDX: VanEck Vectors Gold Miners ETF
gdx_data = yf.download('GDX', start_date, end_date, auto_adjust=True)
data['GDX_Close'] = gdx_data['Close']
feature_list.append('GDX_Close')

# USO: United States Oil Fund ETF
uso_data = yf.download('USO', start_date, end_date, auto_adjust=True)
data['USO_Close'] = uso_data['Close']
feature_list.append('USO_Close')

# 10-Year Treasury Yield (TNX)
treasury_yield_data = yf.download('^TNX', start_date, end_date, auto_adjust=True)
data['10Y_Treasury_Yield'] = treasury_yield_data['Close']
feature_list.append('10Y_Treasury_Yield')

# DXY: US Dollar Index
dxy_data = yf.download('DX-Y.NYB', start_date, end_date, auto_adjust=True)
data['DXY_Close'] = dxy_data['Close']
feature_list.append('DXY_Close')

# Drop any remaining rows with NaN values that may have resulted from the feature engineering process
data = data.dropna()