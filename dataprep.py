# Import the required libraries
import numpy as np
import yfinance as yf
from ta import momentum, trend, volatility, volume
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings("ignore")

# Set the date range for downloading historical data
start_date = '2008-01-01'
end_date = '2024-08-01'

# Download the historical data for the GLD ETF (which tracks gold prices)
# Auto-adjust is used to adjust the data for dividends and stock splits
data = yf.download('GLD', start_date, end_date, auto_adjust=True)

# Drop any rows with missing values to ensure clean data for analysis
data = data.dropna()

# Initialize a list to keep track of the names of the generated features
feature_list = []

# Generate technical indicators for different time windows (5, 10, ..., 50 days)
for i in range(5, 51, 5):
    # SMA (Simple Moving Average): Average of the closing prices over the window
    feature_name = 'SMA' + str(i)
    feature_list.append(feature_name)
    data[feature_name] = trend.sma_indicator(data['Close'], i, fillna=False)

    # EMA (Exponential Moving Average): Weighted moving average giving more weight to recent data
    feature_name = 'EMA' + str(i)
    feature_list.append(feature_name)
    data[feature_name] = trend.ema_indicator(data['Close'], i, fillna=False)

    # ATR (Average True Range): Measures market volatility by considering recent price range
    feature_name = 'ATR' + str(i)
    feature_list.append(feature_name)
    data[feature_name] = volatility.average_true_range(data['High'], data['Low'], data['Close'], i, fillna=False)

    # ADX (Average Directional Index): Measures the strength of a trend
    feature_name = 'ADX' + str(i)
    feature_list.append(feature_name)
    data[feature_name] = trend.adx(data['High'], data['Low'], data['Close'], i, fillna=False)

    # CCI (Commodity Channel Index): Measures the deviation of the price from its average
    feature_name = 'CCI' + str(i)
    feature_list.append(feature_name)
    data[feature_name] = trend.cci(data['High'], data['Low'], data['Close'], i, fillna=False)

    # ROC (Rate of Change): Measures the percentage change in price over a specified time period
    feature_name = 'ROC' + str(i)
    feature_list.append(feature_name)
    data[feature_name] = momentum.roc(data['Close'], i, fillna=False)

    # RSI (Relative Strength Index): Measures the magnitude of recent price changes to evaluate overbought or oversold conditions
    feature_name = 'RSI' + str(i)
    feature_list.append(feature_name)
    data[feature_name] = momentum.rsi(data['Close'], i, fillna=False)

    # Force Index: Combines price and volume to identify the strength of market moves
    feature_name = 'ForceIndex' + str(i)
    feature_list.append(feature_name)
    data[feature_name] = volume.force_index(data['Close'], data['Volume'], i, fillna=False)

    # MFI (Money Flow Index): Volume-weighted version of the RSI, indicates overbought or oversold conditions
    feature_name = 'MFI' + str(i)
    feature_list.append(feature_name)
    data[feature_name] = volume.money_flow_index(data['High'], data['Low'], data['Close'], data['Volume'], i, fillna=False)

    # OBV (On Balance Volume): Measures cumulative buying and selling pressure based on volume
    feature_name = 'OBV' + str(i)
    feature_list.append(feature_name)
    data[feature_name] = volume.on_balance_volume(data['Close'], data['Volume'], i)

    # Ulcer Index (UI): Measures downside risk or volatility in comparison to a previous high
    feature_name = 'UI' + str(i)
    feature_list.append(feature_name)
    data[feature_name] = volatility.ulcer_index(data['Close'], i, fillna=False)


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

# Shift the close prices to create the target variable for supervised learning
data['next_day_price'] = data['Close'].shift(-1)
data = data.dropna() # Drop NaN values created by the shift



# CORRELATION-BASED FEATURE SELECTION :
# Create a correlation matrix to identify features strongly correlated with the target
threshold = 0.7 #  Threshold for correlation 
corr = data.corr()

# Select features with absolute correlation above the threshold
corr_features = corr['next_day_price'].abs()[lambda x: x > threshold].index.difference(['next_day_price']).tolist()


# F-REGRESSION TEST BASED FEATURE SELECTION :
# Using SelectKBest to select the top 15 features based on F-regression test
X = data.drop(columns=['next_day_price'])  # Features (dropping the target)
y = data['next_day_price'].values  # Target variable

# Select top 15 features based on F-regression test
fs = SelectKBest(score_func=f_regression, k=15)
X_selected = fs.fit_transform(X, y)

# Get the names of the selected features
Freg_features = X.columns[fs.get_support()].tolist()


# RANDOM FOREST FEATURE SELECTION :
# Fit a RandomForestRegressor and use SelectFromModel to identify important features
sel = SelectFromModel(RandomForestRegressor(n_estimators=500, random_state=42))  # Added random_state for reproducibility
sel.fit(X, y)

# Get the names of the selected features
RF_features = X.columns[sel.get_support()].tolist()


# LASSO FEATURE SELECTION :
# Split the data into training and testing sets for model selection with Lasso
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define a pipeline for feature scaling and Lasso regression
pipeline = Pipeline([
    ('scaler', MinMaxScaler()), # MinMaxScaler for feature scaling
    ('model', Lasso()) # Lasso for feature selection 
])

# Perform GridSearch to find the optimal alpha for Lasso
search = GridSearchCV(pipeline,
                      {'model__alpha': np.arange(0.1, 10, 0.1)},
                      scoring="neg_mean_squared_error",
                      cv=5  # Using 5-fold cross-validation
                      )

# Fit the pipeline with the training data
search.fit(X_train, y_train)

# Get the best Lasso model from GridSearch
best_model = search.best_estimator_.named_steps['model']

# Get the features selected by Lasso (non-zero coefficients)
selection_mask = best_model.coef_ != 0
Lasso_features = X.columns[selection_mask].tolist()


# COMBINE SELECTED FEATURES FROM ALL METHODS :
selected_features = list(set(corr_features + Freg_features + RF_features + Lasso_features))

# Reduce the dataset to only include the selected features
data = data[selected_features]

# Save the prepared data to a CSV file for future use
data.to_csv('gold_data.csv')