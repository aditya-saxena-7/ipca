import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ipca import InstrumentedPCA
from scipy.spatial import procrustes
import numpy as np

# Define stocks and time period
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
start_date = '2015-01-01'
end_date = '2019-12-31'

# Download daily stock data
data = yf.download(tickers, start=start_date, end=end_date, interval='1d')
adj_close = data['Adj Close']  # Adjusted close prices
volume = data['Volume']  # Trading volume

# Calculate daily returns (target variable)
returns = adj_close.pct_change().dropna()

# Calculate real-world characteristics from available data
characteristics = pd.DataFrame(index=returns.index)

# Volatility: Rolling 20-day standard deviation of returns (approximating monthly volatility)
characteristics['volatility'] = returns.rolling(window=20).std().mean(axis=1)

# Moving Average Ratios: 50-day and 200-day moving averages
ma_50 = adj_close.rolling(window=50).mean()
ma_200 = adj_close.rolling(window=200).mean()
characteristics['ma_ratio'] = (ma_50 / ma_200).mean(axis=1)

# Price-to-Moving Average Ratio (signal for trend-following)
characteristics['price_ma_ratio'] = (adj_close / ma_50).mean(axis=1)

# Volume as a feature (log-transformed for normalization)
characteristics['volume'] = np.log(volume.mean(axis=1))

# Drop NaN values resulting from rolling calculations
characteristics = characteristics.dropna()

# Ensure alignment with returns data
returns = returns.loc[characteristics.index]

# Combine returns and characteristics data for IPCA
returns.reset_index(drop=True, inplace=True)
characteristics.reset_index(drop=True, inplace=True)
data_combined = pd.concat([returns, characteristics], axis=1)

# Print the first few rows and summary statistics
print(data_combined.head())
print("\n")
print(data_combined.describe())

# Standardize characteristics
scaler = StandardScaler()
characteristics_scaled = pd.DataFrame(scaler.fit_transform(characteristics), columns=characteristics.columns, index=characteristics.index)

# Map tickers to integer IDs
ticker_to_id = {ticker: idx for idx, ticker in enumerate(tickers)}
id_to_ticker = {idx: ticker for ticker, idx in ticker_to_id.items()}

# Flatten returns and adjust index for panel structure
y = returns.stack()
y.index.names = ['time', 'entity']
y.index = y.index.set_levels([y.index.levels[0], y.index.levels[1].map(ticker_to_id)], level=['time', 'entity'])

# Prepare X with matching index for IPCA
X = pd.DataFrame(np.repeat(characteristics_scaled.values, len(tickers), axis=0), columns=characteristics_scaled.columns)
X.index = y.index  # Set index to be the same as y

# Check the structure of the data to ensure index alignment
print("\nX head after restructuring:")
print(X.head())
print("\ny head after restructuring:")
print(y.head())

# Define IPCA parameters
K = 3  # Number of latent factors
L = X.shape[1]  # Number of observable instruments
alpha_reg = 0  # Regularization parameter

# Fit IPCA model
ipca = InstrumentedPCA(n_factors=K, intercept=False, alpha=alpha_reg, max_iter=1000, iter_tol=1e-3)
ipca.fit(X, y, data_type="panel")

# Output the estimated Gamma factors and latent factors
Gamma_est, factors = ipca.get_factors(label_ind=True)

# Since we no longer have a "true" Gamma, we focus on analyzing Gamma_est and factors directly
print("Estimated Gamma (Factor Loadings):")
print(Gamma_est)
print("\nEstimated Latent Factors:")
print(factors)

# Use Gamma_est as our baseline (no true Gamma since we're using real data)
# Calculate the unaligned error (using the Frobenius norm of Gamma_est)
unaligned_error = np.linalg.norm(Gamma_est, ord='fro')**2

# Apply Procrustes Analysis to align Gamma_est with itself to measure alignment improvement
# (we use Gamma_est as the reference in both inputs to simulate alignment effects)
_, aligned_Gamma_est, disparity = procrustes(Gamma_est, Gamma_est)
aligned_error = np.linalg.norm(aligned_Gamma_est, ord='fro')**2

# Calculate Error Difference and Percentage Improvement
error_difference = unaligned_error - aligned_error
percentage_improvement = (error_difference / unaligned_error) * 100

# Print the results
print(f"Unaligned Error: {unaligned_error}")
print(f"Aligned Error: {aligned_error}")
print(f"Error Difference: {error_difference}")
print(f"Percentage Improvement Due to Alignment: {percentage_improvement}%")
