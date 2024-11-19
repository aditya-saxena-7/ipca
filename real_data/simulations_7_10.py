import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ipca import InstrumentedPCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.spatial import procrustes

# Set seaborn style
sns.set(style='whitegrid')

# Define stocks and time period
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
start_date = '2015-01-01'
end_date = '2019-12-31'

# Create a directory for plots
plot_dir = "plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

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

# Cross-validation with reduced factors
kf = KFold(n_splits=5, shuffle=False)
factor_counts = [1, 2]  # Number of factors to test

for n_factors in factor_counts:
    print(f"\nRunning IPCA with {n_factors} factor(s)...")
    Gamma_matrices = []
    unaligned_error_matrix = np.zeros((5, 5))
    aligned_error_matrix = np.zeros((5, 5))
    percentage_diff_matrix = np.zeros((5, 5))

    for fold_idx, (train_index, test_index) in enumerate(kf.split(time_index), start=1):
        # Get train time indices
        train_time_indices = time_index[train_index]
        
        # Create mask for training data
        train_mask = np.isin(indices[:, 0], train_time_indices)
        
        # Split data
        X_train = X[train_mask]
        y_train = y[train_mask]
        indices_train = indices[train_mask]

        # Fit IPCA
        ipca = InstrumentedPCA(
            n_factors=n_factors,  # Number of factors
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

    # Calculate alignment errors
    for i in range(5):
        for j in range(5):
            if i == j:
                # No error when comparing the same fold
                unaligned_error_matrix[i, j] = 0
                aligned_error_matrix[i, j] = 0
                percentage_diff_matrix[i, j] = 0
            else:
                Gamma_i = Gamma_matrices[i].values
                Gamma_j = Gamma_matrices[j].values

                # Unaligned error
                unaligned_error = np.linalg.norm(Gamma_i - Gamma_j, ord='fro')**2
                unaligned_error_matrix[i, j] = unaligned_error

                # Procrustes alignment
                _, aligned_Gamma_j, _ = procrustes(Gamma_i, Gamma_j)
                aligned_error = np.linalg.norm(Gamma_i - aligned_Gamma_j, ord='fro')**2
                aligned_error_matrix[i, j] = aligned_error

                # Percentage improvement
                if unaligned_error != 0:
                    percentage_diff_matrix[i, j] = ((unaligned_error - aligned_error) / unaligned_error) * 100

    # Plot unaligned error heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(unaligned_error_matrix, annot=True, cmap='Reds', xticklabels=[f'Fold {i+1}' for i in range(5)], yticklabels=[f'Fold {i+1}' for i in range(5)])
    plt.title(f'Unaligned Error Heatmap ({n_factors} Factor(s))')
    plt.xlabel('Fold')
    plt.ylabel('Fold')
    plt.savefig(f'{plot_dir}/unaligned_error_heatmap_{n_factors}_factors.png')
    plt.close()

    # Plot aligned error heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(aligned_error_matrix, annot=True, cmap='Blues', xticklabels=[f'Fold {i+1}' for i in range(5)], yticklabels=[f'Fold {i+1}' for i in range(5)])
    plt.title(f'Aligned Error Heatmap ({n_factors} Factor(s))')
    plt.xlabel('Fold')
    plt.ylabel('Fold')
    plt.savefig(f'{plot_dir}/aligned_error_heatmap_{n_factors}_factors.png')
    plt.close()

    # Plot percentage difference heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(percentage_diff_matrix, annot=True, cmap='Greens', xticklabels=[f'Fold {i+1}' for i in range(5)], yticklabels=[f'Fold {i+1}' for i in range(5)])
    plt.title(f'Percentage Difference Heatmap (Unaligned - Aligned) ({n_factors} Factor(s))')
    plt.xlabel('Fold')
    plt.ylabel('Fold')
    plt.savefig(f'{plot_dir}/percentage_diff_heatmap_{n_factors}_factors.png')
    plt.close()

print(f"\nAll results and plots are saved in the '{plot_dir}' directory.")
