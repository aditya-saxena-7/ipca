import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ipca import InstrumentedPCA
from scipy.spatial import procrustes
import numpy as np
from sklearn.model_selection import KFold

# Define stocks and time period
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
start_date = '2015-01-01'
end_date = '2019-12-31'

# Download daily stock data
data = yf.download(tickers, start=start_date, end=end_date, interval='1d')
adj_close = data['Adj Close']
volume = data['Volume']

# Calculate daily returns
returns = adj_close.pct_change().dropna()

# Calculate characteristics
characteristics = pd.DataFrame(index=returns.index)
characteristics['volatility'] = returns.rolling(window=20).std().mean(axis=1)
ma_50 = adj_close.rolling(window=50).mean()
ma_200 = adj_close.rolling(window=200).mean()
characteristics['ma_ratio'] = (ma_50 / ma_200).mean(axis=1)
characteristics['price_ma_ratio'] = (adj_close / ma_50).mean(axis=1)
characteristics['volume'] = np.log(volume.mean(axis=1))

# Drop NaN values and align data
characteristics = characteristics.dropna()
returns = returns.loc[characteristics.index]

# Standardize characteristics
scaler = StandardScaler()
characteristics_scaled = pd.DataFrame(
    scaler.fit_transform(characteristics),
    columns=characteristics.columns,
    index=characteristics.index
)

# Create panel data structure
time_index = np.arange(len(characteristics_scaled))
entity_index = np.arange(len(tickers))
n_time = len(time_index)
n_entity = len(entity_index)

# Create indices array for the panel structure
indices = np.array([
    [t, e] for t in time_index for e in entity_index
])

# Prepare X (characteristics matrix)
X = np.repeat(characteristics_scaled.values, n_entity, axis=0)

# Prepare y (returns vector)
y = returns.values.reshape(-1, order='F')  # Reshape to match panel structure

# Cross-validation
kf = KFold(n_splits=5, shuffle=False)
Gamma_matrices = []

print(f"Data dimensions:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"indices shape: {indices.shape}")

# Run IPCA on each fold
for fold_idx, (train_index, test_index) in enumerate(kf.split(time_index), start=1):
    # Get train time indices
    train_time_indices = time_index[train_index]
    
    # Create mask for training data
    train_mask = np.isin(indices[:, 0], train_time_indices)
    
    # Split data
    X_train = X[train_mask]
    y_train = y[train_mask]
    indices_train = indices[train_mask]
    
    print(f"\nFold {fold_idx} - Training data dimensions:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"indices_train shape: {indices_train.shape}")
    
    # Fit IPCA
    ipca = InstrumentedPCA(
        n_factors=3,  # Number of factors
        intercept=False,
        alpha=0,  # No regularization
        max_iter=1000,
        iter_tol=1e-3
    )
    
    ipca.fit(
        X_train,
        y_train,
        data_type="panel",
        indices=indices_train
    )
    
    # Get estimated parameters
    Gamma_est, factors = ipca.get_factors(label_ind=True)
    Gamma_matrices.append(Gamma_est)
    
    print(f"\nFold {fold_idx} - Estimated Gamma (Factor Loadings):")
    print(Gamma_est)
    print(f"\nEstimated Latent Factors shape for Fold {fold_idx}:")
    print(factors.shape)

# Procrustes analysis
if len(Gamma_matrices) > 1:
    reference_Gamma = Gamma_matrices[0]
    aligned_errors = []
    unaligned_errors = []

    for i, Gamma_est in enumerate(Gamma_matrices[1:], start=1):
        # Calculate unaligned error
        unaligned_error = np.linalg.norm(reference_Gamma - Gamma_est, ord='fro')**2
        unaligned_errors.append(unaligned_error)
        
        # Apply Procrustes alignment
        _, aligned_Gamma, _ = procrustes(reference_Gamma, Gamma_est)
        aligned_error = np.linalg.norm(reference_Gamma - aligned_Gamma, ord='fro')**2
        aligned_errors.append(aligned_error)
        
        print(f"\nAlignment Results for Fold {i}:")
        print(f"Unaligned Error: {unaligned_error:.6f}")
        print(f"Aligned Error: {aligned_error:.6f}")
        print(f"Improvement: {((unaligned_error - aligned_error) / unaligned_error * 100):.2f}%")

    # Summary statistics
    print("\nCross-Validation Summary:")
    print(f"Average Unaligned Error: {np.mean(unaligned_errors):.6f}")
    print(f"Average Aligned Error: {np.mean(aligned_errors):.6f}")
    print(f"Average Improvement: {((np.mean(unaligned_errors) - np.mean(aligned_errors)) / np.mean(unaligned_errors) * 100):.2f}%")