import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from ipca import InstrumentedPCA  # Ensure this is installed: `pip install ipca`
import pandas_datareader.data as web
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Step 1: Load Fama-French 5-Factor data
ff_factors = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2015', end='2020')[0]

# Convert the PeriodIndex to Timestamp and reset index so that Date is a column
ff_factors.index = ff_factors.index.to_timestamp()
ff_factors.reset_index(inplace=True)  # Reset index so Date becomes a column
ff_factors = ff_factors.rename(columns={'Mkt-RF': 'Market_Risk_Premium', 'SMB': 'Size_Premium', 'HML': 'Value_Premium',
                                        'RMW': 'Profitability_Premium', 'CMA': 'Investment_Premium'})
ff_factors['Market_Returns'] = ff_factors['Market_Risk_Premium'] + ff_factors['RF']

# Step 2: Download stock data for a list of companies
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NFLX', 'NVDA', 'ORCL', 'IBM']
stock_data = yf.download(tickers, start='2015-01-01', end='2020-01-01', interval='1mo')['Adj Close'].pct_change().dropna()

# Step 3: Remove timezone info
stock_data.index = pd.to_datetime(stock_data.index).tz_localize(None)

# Reshape stock data into long format
stock_data['Date'] = stock_data.index
stock_data = pd.melt(stock_data, id_vars=['Date'], var_name='Firm', value_name='Stock_Returns')

# Step 4: Ensure 'Date' column is in the same format in both DataFrames
ff_factors['Date'] = pd.to_datetime(ff_factors['Date'])

# Step 5: Merge stock returns with Fama-French factors on the 'Date' column
data = pd.merge(stock_data, ff_factors, on='Date')

# **Map 'Firm' names to integer IDs**
firm_names = data['Firm'].unique()
firm_id_map = {name: idx for idx, name in enumerate(firm_names)}
data['Firm_ID'] = data['Firm'].map(firm_id_map)

# **Map 'Date' values to integer IDs**
date_values = data['Date'].unique()
date_id_map = {date: idx for idx, date in enumerate(date_values)}
data['Date_ID'] = data['Date'].map(date_id_map)

# **Reset and set new index**
data.reset_index(drop=True, inplace=True)
data.set_index(['Firm_ID', 'Date_ID'], inplace=True)

# Prepare data for IPCA
# Dependent variable: Stock_Returns
y = data['Stock_Returns']

# Independent variables
X = data[['Market_Risk_Premium', 'Size_Premium', 'Value_Premium',
          'Profitability_Premium', 'Investment_Premium', 'Market_Returns']]

# **Drop 'Market_Returns' from X to reduce multicollinearity**
X = X.drop(columns=['Market_Returns'])

# Extract indices as integer arrays
indices = np.array(list(zip(data.index.get_level_values('Firm_ID'), data.index.get_level_values('Date_ID'))))

# **Proceed with train_test_split including indices**
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X, y, indices, test_size=0.2, random_state=42
)

# **Standardize the features**
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert X_train_scaled back to DataFrame if it's not
X_vif = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]

print(vif_data)

print("X_train_scaled shape:", X_train_scaled.shape)
print("y_train shape:", y_train.shape)
print("indices_train shape:", indices_train.shape)
print("Indices data type:", indices_train.dtype)

# **Convert scaled arrays back to DataFrames to retain indices**
X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

# **Check for NaNs and infinite values**
print("NaNs in X_train_scaled:", X_train_scaled.isnull().any())
print("NaNs in y_train:", y_train.isnull().any())

# **Fit the IPCA model providing indices**
ipca = InstrumentedPCA(n_factors=1, intercept=False, alpha=1e6)
ipca_model = ipca.fit(X=X_train_scaled, y=y_train, indices=indices_train, data_type='panel')

# **Get estimated factors and loadings**
Gamma, Factors = ipca.get_factors(label_ind=True)

# **Display the Gamma (loading matrix) and the estimated Factors**
print(f"Gamma (Loading Matrix): \n{Gamma}")
print(f"Factors: \n{Factors}")
