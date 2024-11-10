import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ipca import InstrumentedPCA
from scipy.spatial import procrustes
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style='whitegrid')

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

# Print descriptive statistics of returns
print("Descriptive Statistics of Daily Returns:")
print(returns.describe())
print("\n")

# Plot histograms of returns
for ticker in tickers:
    plt.figure(figsize=(8, 6))
    sns.histplot(returns[ticker], bins=50, kde=True)
    plt.title(f'Distribution of Daily Returns for {ticker}')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.savefig(f'{ticker}_returns_histogram.png')
    plt.close()

# Plot time series of returns
for ticker in tickers:
    plt.figure(figsize=(12, 6))
    plt.plot(returns.index, returns[ticker])
    plt.title(f'Daily Returns Over Time for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.savefig(f'{ticker}_returns_timeseries.png')
    plt.close()

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

# Print descriptive statistics of characteristics
print("Descriptive Statistics of Characteristics:")
print(characteristics.describe())
print("\n")

# Plot correlation heatmap of characteristics
plt.figure(figsize=(8, 6))
sns.heatmap(characteristics.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Characteristics')
plt.savefig('characteristics_correlation_heatmap.png')
plt.close()

# Pairplot of characteristics
sns.pairplot(characteristics)
plt.savefig('characteristics_pairplot.png')
plt.close()

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
variance_explained_list = []

print(f"Data Dimensions:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"indices shape: {indices.shape}")

# Run IPCA on each fold
fold_numbers = []
for fold_idx, (train_index, test_index) in enumerate(kf.split(time_index), start=1):
    # Get train time indices
    train_time_indices = time_index[train_index]

    # Create mask for training data
    train_mask = np.isin(indices[:, 0], train_time_indices)

    # Split data
    X_train = X[train_mask]
    y_train = y[train_mask]
    indices_train = indices[train_mask]

    print(f"\nFold {fold_idx} - Training Data Dimensions:")
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

    # Print initial convergence parameters
    print(f"\nFold {fold_idx} - Fitting IPCA Model...")

    ipca.fit(
        X_train,
        y_train,
        data_type="panel",
        indices=indices_train
    )

    # Get estimated parameters
    Gamma_est, factors = ipca.get_factors(label_ind=True)
    Gamma_matrices.append(Gamma_est)
    fold_numbers.append(f'Fold {fold_idx}')

    # Since 'n_iter_' and 'aggregate_update_' are not available, we can't print them

    # Print estimated Gamma
    print(f"\nFold {fold_idx} - Estimated Gamma (Factor Loadings):")
    print(Gamma_est)
    print(f"\nEstimated Latent Factors for Fold {fold_idx}:")
    print(factors)
    print(f"Factors Shape: {factors.shape}")

    # Additional statistics
    # Compute variance explained by factors
    factor_covariance = np.cov(factors)
    total_variance = np.var(y_train)
    explained_variance = np.trace(factor_covariance)
    variance_explained = explained_variance / total_variance * 100
    variance_explained_list.append(variance_explained)
    print(f"\nFold {fold_idx} - Variance Explained by Factors: {explained_variance:.6f}")
    print(f"Total Variance in y_train: {total_variance:.6f}")
    print(f"Proportion of Variance Explained: {variance_explained:.2f}%")

    # Plot heatmap of Gamma_est
    plt.figure(figsize=(8, 6))
    sns.heatmap(Gamma_est, annot=True, cmap='viridis')
    plt.title(f'Gamma Matrix Heatmap - Fold {fold_idx}')
    plt.xlabel('Factors')
    plt.ylabel('Characteristics')
    plt.savefig(f'gamma_heatmap_fold_{fold_idx}.png')
    plt.close()

# Procrustes analysis
if len(Gamma_matrices) > 1:
    reference_Gamma = Gamma_matrices[0].values
    aligned_errors = []
    unaligned_errors = []
    error_differences = []

    n_folds = len(Gamma_matrices)
    unaligned_error_matrix = np.zeros((n_folds, n_folds))
    aligned_error_matrix = np.zeros((n_folds, n_folds))
    error_difference_matrix = np.zeros((n_folds, n_folds))

    for i in range(n_folds):
        for j in range(n_folds):
            if i == j:
                unaligned_error = 0
                aligned_error = 0
            else:
                Gamma_i = Gamma_matrices[i].values
                Gamma_j = Gamma_matrices[j].values

                # Calculate unaligned error
                unaligned_error = np.linalg.norm(Gamma_i - Gamma_j, ord='fro')**2

                # Apply Procrustes alignment
                _, aligned_Gamma_j, _ = procrustes(Gamma_i, Gamma_j)
                aligned_error = np.linalg.norm(Gamma_i - aligned_Gamma_j, ord='fro')**2

                # Store errors
                unaligned_errors.append(unaligned_error)
                aligned_errors.append(aligned_error)
                error_differences.append(unaligned_error - aligned_error)

            unaligned_error_matrix[i, j] = unaligned_error
            aligned_error_matrix[i, j] = aligned_error
            error_difference_matrix[i, j] = unaligned_error - aligned_error

    # Plot Unaligned Error Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(unaligned_error_matrix, annot=True, cmap='Reds', xticklabels=fold_numbers, yticklabels=fold_numbers)
    plt.title('Unaligned Error Heatmap')
    plt.xlabel('Fold')
    plt.ylabel('Fold')
    plt.savefig('unaligned_error_heatmap.png')
    plt.close()

    # Plot Aligned Error Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(aligned_error_matrix, annot=True, cmap='Blues', xticklabels=fold_numbers, yticklabels=fold_numbers)
    plt.title('Aligned Error Heatmap')
    plt.xlabel('Fold')
    plt.ylabel('Fold')
    plt.savefig('aligned_error_heatmap.png')
    plt.close()

    # Plot Error Difference Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(error_difference_matrix, annot=True, cmap='Greens', xticklabels=fold_numbers, yticklabels=fold_numbers)
    plt.title('Error Difference Heatmap (Unaligned - Aligned)')
    plt.xlabel('Fold')
    plt.ylabel('Fold')
    plt.savefig('error_difference_heatmap.png')
    plt.close()

    # Alignment Comparison Plot
    error_df = pd.DataFrame({
        'Unaligned Error': unaligned_errors,
        'Aligned Error': aligned_errors,
        'Improvement (%)': [(u - a) / u * 100 if u != 0 else 0 for u, a in zip(unaligned_errors, aligned_errors)]
    })

    plt.figure(figsize=(10, 6))
    error_df[['Unaligned Error', 'Aligned Error']].plot(kind='bar')
    plt.title('Alignment Comparison')
    plt.xlabel('Error Pair Index')
    plt.ylabel('Error')
    plt.savefig('alignment_comparison.png')
    plt.close()

    # Summary statistics
    avg_unaligned_error = np.mean(unaligned_errors)
    avg_aligned_error = np.mean(aligned_errors)
    avg_improvement = ((avg_unaligned_error - avg_aligned_error) / avg_unaligned_error * 100)

    print("\nCross-Validation Summary:")
    print(f"Average Unaligned Error: {avg_unaligned_error:.6f}")
    print(f"Average Aligned Error: {avg_aligned_error:.6f}")
    print(f"Average Improvement: {avg_improvement:.2f}%")

    # Variance Explained Plot
    plt.figure(figsize=(8, 6))
    plt.bar(fold_numbers, variance_explained_list)
    plt.title('Variance Explained by Factors in Each Fold')
    plt.xlabel('Fold')
    plt.ylabel('Variance Explained (%)')
    plt.savefig('variance_explained.png')
    plt.close()

print("\nAll plots have been saved to the current directory.")
