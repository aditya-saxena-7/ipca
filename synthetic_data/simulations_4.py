import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipca import InstrumentedPCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler

''' ### Issues

The warning you are encountering indicates that the objective function in the **Elastic Net** algorithm (used in the IPCA model) is not converging properly. The `ConvergenceWarning` in `sklearn.linear_model._coordinate_descent` suggests a few potential issues that could be addressed:

### Causes of the Issue:
1. **Number of Iterations**: The default maximum number of iterations may be insufficient for the model to converge. Increasing the number of iterations may help.
   
2. **Feature Scaling**: If the features in the dataset have very different scales, the optimization process may struggle to converge. Standardizing or normalizing the input features could improve convergence.
   
3. **Regularization Strength**: The warning suggests that the regularization may not be strong enough to help the model converge. Increasing the regularization parameter \(\alpha\) might resolve this issue.

### Steps to Address the Warning:

#### 1. **Increase the Number of Iterations**:
You can increase the maximum number of iterations in the IPCA model to give the algorithm more time to converge.

#### 2. **Scale the Features**:
If the features (observable instruments \(c_{i,t}\)) have different ranges, normalizing them could help with convergence. You can scale the features using `StandardScaler` from `sklearn`.

#### 3. **Increase Regularization**:
Increasing the `alpha` parameter adds more regularization, which can help the model converge, especially when dealing with noisy or poorly scaled data.

### Summary of Adjustments:
1. **Increased the number of iterations**: Set `max_iter=1000` to give the model more time to converge.
2. **Standardized the features**: Used `StandardScaler` to scale the input features so that they are on a similar scale, helping with convergence.
3. **Increased regularization strength**: Set `alpha=1e-3` to add more regularization, which can stabilize the optimization and improve convergence.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ipca import InstrumentedPCA
import seaborn as sns

# Simulation Parameters
T = 200  # Number of time periods
K = 3    # Number of latent factors
L = 5    # Number of observable instruments
alpha_reg = 1e3  # Reduced regularization to avoid over-shrinking
max_iterations = 10000  # Maximum number of iterations
tolerance = 1e-3  # Convergence tolerance
sample_sizes = [50, 100, 200, 500, 1000]  # Avoid small sample sizes to prevent matrix issues
sigma_squares = np.logspace(-2, 0, 10)  # Values for sigma^2 ranging from 0.01 to 1

# Function to simulate data
def simulate_data(N, T, K, L, Gamma_true, sigma_e):
    # Generate latent factors
    A = np.array([[0.8, 0.1, 0.05],
                  [0.1, 0.85, 0.05],
                  [0.05, 0.1, 0.9]])  # Transition matrix for factors
    f_t = np.zeros((T, K))  # Latent factors
    for t in range(1, T):
        eta_t = np.random.normal(0, 0.1, K)  # Noise term
        f_t[t] = A @ f_t[t-1] + eta_t

    # Generate instruments
    B = np.array([[0.7, 0.1, 0.05, 0.05, 0.1],
                  [0.05, 0.8, 0.1, 0.05, 0.05],
                  [0.05, 0.05, 0.75, 0.1, 0.05],
                  [0.1, 0.05, 0.1, 0.85, 0.05],
                  [0.05, 0.05, 0.1, 0.05, 0.8]])  # Transition matrix for instruments
    c_it = np.zeros((N, T, L))
    for i in range(N):
        for t in range(1, T):
            eps_it = np.random.normal(0, 0.1, L)  # Noise term for instruments
            c_it[i, t] = B @ c_it[i, t-1] + eps_it

    # Generate errors and observed data
    e_it = np.random.normal(0, sigma_e, (N, T))  # Error term
    x_it = np.zeros((N, T))
    for i in range(N):
        for t in range(T):
            x_it[i, t] = c_it[i, t] @ Gamma_true @ f_t[t] + e_it[i, t]

    return x_it, f_t, c_it

# Ensure matrix is positive definite by adding a small positive value to the diagonal
def ensure_positive_definite(matrix):
    epsilon = 1e-6  # Small value added to the diagonal
    return matrix + epsilon * np.eye(matrix.shape[0])

# Initialize true Gamma (factor loadings matrix)
Gamma_true = np.random.normal(0, 0.1, (L, K))

# Initialize StandardScaler for feature scaling
scaler = StandardScaler()

# Store errors for different combinations of sigma^2 and N
errors = []

for sigma_e in sigma_squares:
    for N in sample_sizes:
        # Simulate data
        X, f_t, c_it = simulate_data(N, T, K, L, Gamma_true, sigma_e)
        
        # Reshape data for IPCA
        X_flat = X.flatten()
        c_it_flat = c_it.reshape(N * T, L)  # Reshape instruments
        
        # Standardize the features (instruments)
        c_it_flat_scaled = scaler.fit_transform(c_it_flat)

        # Prepare for IPCA
        y_flat = X_flat  # Observed data
        entities = np.repeat(np.arange(N), T)  # Entity index
        times = np.tile(np.arange(T), N)  # Time index
        index = pd.MultiIndex.from_arrays([entities, times], names=['entity', 'time'])
        
        # Convert instruments and data into pandas DataFrame and Series with MultiIndex
        c_it_df_scaled = pd.DataFrame(c_it_flat_scaled, index=index)
        y_df = pd.Series(y_flat, index=index)
        
        # Fit the IPCA model with adjusted parameters
        regr = InstrumentedPCA(n_factors=K, intercept=False, alpha=alpha_reg, max_iter=max_iterations, l1_ratio=0.001, iter_tol=tolerance, n_jobs=-1)
        try:
            regr = regr.fit(X=c_it_df_scaled, y=y_df)
        except np.linalg.LinAlgError as e:
            print(f"LinAlgError encountered for N={N}, sigma^2={sigma_e}: {e}")
            continue  # Skip this iteration

        # Retrieve the estimated Gamma (factor loadings)
        Gamma_est, _ = regr.get_factors(label_ind=True)
        
        # Calculate the error ||Gamma - Gamma*||^2
        error = np.linalg.norm(Gamma_true - Gamma_est, ord='fro')**2
        
        # Store error with corresponding sigma^2 and N
        errors.append([sigma_e, N, error])

# Convert errors to DataFrame for easier plotting
df_errors = pd.DataFrame(errors, columns=['sigma^2', 'N', 'Error'])

# Plotting the result using a heatmap to visualize the relationship between sigma^2, N, and Error
plt.figure(figsize=(10, 8))
pivot_table = df_errors.pivot("sigma^2", "N", "Error")
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis")
plt.title(r'$\| \Gamma - \Gamma^* \|^2$ for different $\sigma^2$ and N')
plt.xlabel('Sample Size (N)')
plt.ylabel(r'$\sigma^2$ (Noise Variance)')
plt.show()
