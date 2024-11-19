import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ipca import InstrumentedPCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
results = []

for n_factors in factor_counts:
    print(f"\nRunning IPCA with {n_factors} factor(s)...")
    fold_results = []
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

        # Compute variance explained
        if n_factors == 1:
            factor_variance = np.var(factors.values.flatten(), ddof=1)
            explained_variance = factor_variance
        else:
            factor_covariance = np.cov(factors)
            explained_variance = np.trace(factor_covariance)

        total_variance = np.var(y_train, ddof=1)
        variance_explained = explained_variance / total_variance * 100
        fold_results.append(variance_explained)

        # Save heatmap of Gamma_est
        plt.figure(figsize=(8, 6))
        sns.heatmap(Gamma_est, annot=True, cmap='viridis')
        plt.title(f'Gamma Matrix Heatmap - Fold {fold_idx} ({n_factors} Factor(s))')
        plt.xlabel('Factors')
        plt.ylabel('Characteristics')
        plt.savefig(f'{plot_dir}/gamma_heatmap_fold_{fold_idx}_{n_factors}_factors.png')
        plt.close()

    results.append(fold_results)

# Plot Variance Explained for 1 vs. 2 Factors
plt.figure(figsize=(10, 6))
for i, n_factors in enumerate(factor_counts):
    plt.plot(range(1, 6), results[i], label=f"{n_factors} factor(s)")
plt.title("Variance Explained by Factors Across Folds")
plt.xlabel("Fold")
plt.ylabel("Variance Explained (%)")
plt.legend()
plt.savefig(f'{plot_dir}/variance_explained_comparison.png')
plt.close()

# Summary Statistics for 1 vs. 2 Factors
summary = []
for i, n_factors in enumerate(factor_counts):
    avg_variance = np.mean(results[i])
    summary.append({
        'Factors': n_factors,
        'Avg_Variance_Explained': avg_variance,
        'Variance_Explained_Per_Fold': results[i]
    })
    print(f"\nSummary for {n_factors} Factor(s):")
    print(f"Average Variance Explained Across Folds: {avg_variance:.2f}%")
    print(f"Variance Explained in Each Fold: {results[i]}")

# Save summary to CSV
summary_df = pd.DataFrame(summary)
summary_df.to_csv(f'{plot_dir}/variance_explained_summary.csv', index=False)

print(f"\nAll results and plots are saved in the '{plot_dir}' directory.")
