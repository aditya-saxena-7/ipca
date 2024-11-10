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

# Download stock data (monthly frequency)
data = yf.download(tickers, start=start_date, end=end_date, interval='1d')
data = data['Adj Close']  # Adjusted close prices

# Calculate monthly returns
returns = data.pct_change().dropna()

# Create additional characteristics for each stock and each month
characteristics = pd.DataFrame(index=returns.index)

# Add Market Cap (constant values for demonstration)
characteristics['market_cap'] = data.mean() * 1e7  # Mean adjusted close * 10 million as placeholder

# Add Price-to-Earnings Ratio (PE ratio, hypothetical time-variant)
characteristics['pe_ratio'] = [20 + i * 0.5 for i in range(len(returns))]

# Calculate volatility (rolling 3-month standard deviation of returns)
characteristics['volatility'] = returns.rolling(window=3).std().mean(axis=1)

# Add simulated Dividend Yield (for illustration purposes)
characteristics['div_yield'] = np.random.uniform(1, 3, len(returns))  # Simulated dividend yield in %

# Add Price-to-Book Ratio (simulated time-variant values)
characteristics['pb_ratio'] = np.random.uniform(1, 5, len(returns))  # Price-to-Book Ratio

# Combine returns and characteristics data for IPCA
returns.reset_index(drop=True, inplace=True)
characteristics.reset_index(drop=True, inplace=True)
data_combined = pd.concat([returns, characteristics], axis=1)

# Print the first few rows and summary statistics
print(data_combined.head())
print("\n")
print(data_combined.describe())

# Standardize characteristics
characteristics = characteristics.dropna(axis=1)  # Drop any columns with NaNs (e.g., market_cap may have NaNs)
scaler = StandardScaler()
characteristics_scaled = pd.DataFrame(scaler.fit_transform(characteristics), columns=characteristics.columns, index=characteristics.index)

# Map tickers to integer IDs
ticker_to_id = {ticker: idx for idx, ticker in enumerate(tickers)}
id_to_ticker = {idx: ticker for ticker, idx in ticker_to_id.items()}

# Flatten returns and adjust index
y = returns.stack()
y.index.names = ['time', 'entity']
# Map 'entity' level to integer IDs
y.index = y.index.set_levels([y.index.levels[0], y.index.levels[1].map(ticker_to_id)], level=['time', 'entity'])

# Prepare X with matching index
X = pd.DataFrame(np.repeat(characteristics_scaled.values, len(tickers), axis=0), columns=characteristics_scaled.columns)
X.index = y.index  # Set index to be the same as y

# Check the structure of the data to ensure index alignment
print("\nX head after restructuring:")
print(X.head())
print("\ny head after restructuring:")
print(y.head())
print("\nIndex data types (X):", X.index.levels[0].dtype, X.index.levels[1].dtype)
print("Index data types (y):", y.index.levels[0].dtype, y.index.levels[1].dtype)

# Define IPCA parameters
K = 3  # Number of latent factors
L = X.shape[1]  # Number of observable instruments
alpha_reg = 0  # Regularization parameter

# Fit IPCA model
ipca = InstrumentedPCA(n_factors=K, intercept=False, alpha=alpha_reg, max_iter=1000, iter_tol=1e-3)
ipca.fit(X, y, data_type="panel")

# Estimate Gamma and latent factors
Gamma_true = np.random.normal(0, 0.1, (L, K))  # Example "true" Gamma for error calculation
Gamma_est, factors = ipca.get_factors(label_ind=True)

# Calculate Unaligned Error
unaligned_error = np.linalg.norm(Gamma_true - Gamma_est, ord='fro')**2

# Apply Procrustes Analysis for alignment
_, aligned_Gamma_est, disparity = procrustes(Gamma_true, Gamma_est)
aligned_error = np.linalg.norm(Gamma_true - aligned_Gamma_est, ord='fro')**2

# Calculate Error Difference and Percentage Improvement
error_difference = unaligned_error - aligned_error
percentage_improvement = (error_difference / unaligned_error) * 100

print(f"Unaligned Error: {unaligned_error}")
print(f"Aligned Error: {aligned_error}")
print(f"Error Difference: {error_difference}")
print(f"Percentage Improvement Due to Alignment: {percentage_improvement}%")
