import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import yfinance as yf
from ipca import InstrumentedPCA  # Ensure this is installed: `pip install ipca`
import pandas_datareader.data as web

# Step 1: Load Fama-French 5-Factor data
ff_factors = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2015', end='2020')[0]
print(ff_factors)

# Convert the PeriodIndex to Timestamp and reset index so that Date is a column
ff_factors.index = ff_factors.index.to_timestamp()
ff_factors.reset_index(inplace=True)  # Reset index so Date becomes a column
ff_factors = ff_factors.rename(columns={'Mkt-RF': 'Market_Risk_Premium', 'SMB': 'Size_Premium', 'HML': 'Value_Premium',
                                        'RMW': 'Profitability_Premium', 'CMA': 'Investment_Premium'})
ff_factors['Market_Returns'] = ff_factors['Market_Risk_Premium'] + ff_factors['RF']

# Step 2: Download stock data for a list of companies (Apple, Microsoft, Google, Amazon, Meta)
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NFLX', 'NVDA', 'ORCL', 'IBM']
stock_data = yf.download(tickers, start='2015-01-01', end='2020-01-01', interval='1mo')['Adj Close'].pct_change().dropna()

# Step 3: Make sure the stock data 'Date' column is not in UTC
stock_data.index = pd.to_datetime(stock_data.index).tz_localize(None)  # Remove timezone info

# Reshape stock data into long format
stock_data['Date'] = stock_data.index
stock_data = pd.melt(stock_data, id_vars=['Date'], var_name='Firm', value_name='Stock_Returns')

# Step 4: Ensure 'Date' column is in the same format in both DataFrames
ff_factors['Date'] = pd.to_datetime(ff_factors['Date'])

# Step 5: Merge stock returns with Fama-French factors on the 'Date' column
data = pd.merge(stock_data, ff_factors, on='Date')

# Check the merged data
# Basic information
print(data.info())
print(data.describe())
'''
# Visualize stock returns for different firms
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Date', y='Stock_Returns', hue='Firm', legend=True)
plt.title("Stock Returns Over Time")
plt.show()

# Exclude non-numeric columns ('Firm' and 'Date') for correlation matrix
numeric_data = data.drop(columns=['Firm', 'Date'])

# Correlation matrix of factors and stock returns
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()'''

# **Set the index to be MultiIndex of 'Firm' and 'Date'**
data.set_index(['Firm', 'Date'], inplace=True)

# Prepare data for IPCA
# Dependent variable: Stock_Returns
y = data['Stock_Returns']

# Independent variables: All Fama-French factors and Market_Returns
X = data[['Market_Risk_Premium', 'Size_Premium', 'Value_Premium',
          'Profitability_Premium', 'Investment_Premium', 'Market_Returns']]

# **Proceed with train_test_split without indices**
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Check for multicollinearity
corr_matrix = X_train.corr()
print("Correlation Matrix:\n", corr_matrix)

# **Standardize the features**
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# **Convert scaled arrays back to DataFrames to retain indices**
X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

# **Fit the IPCA model without providing indices**
ipca = InstrumentedPCA(n_factors=1, intercept=True, alpha=1e-3)
ipca_model = ipca.fit(X=X_train_scaled, y=y_train, data_type='panel')

# **Get estimated factors and loadings**
Gamma, Factors = ipca.get_factors(label_ind=True)

# **Display the Gamma (loading matrix) and the estimated Factors**
print(f"Gamma (Loading Matrix): \n{Gamma}")
print(f"Factors: \n{Factors}")
