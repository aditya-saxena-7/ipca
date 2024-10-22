import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipca import InstrumentedPCA

# Simulation Parameters
T = 200  # Number of time periods
K = 3    # Number of latent factors
L = 5    # Number of observable instruments
sigma_e = 0.2  # Standard deviation of the error term
alpha_reg = 1e-4  # Increase regularization for faster convergence
max_iterations = 100  # Reduce max iterations for faster convergence
tolerance = 1e-4  # Increase tolerance to allow early stopping

# Step 1: Generate true Gamma (Gamma*)
Gamma_true = np.random.normal(0, 0.1, (L, K))  # True loadings matrix (L instruments to K factors)

def simulate_data(N, T, K, L, Gamma_true, sigma_e):
    """
    Simulates data for IPCA model.
    Returns: X (observed data), f_t (latent factors), Gamma (true loadings), c_it (instruments)
    """
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

    return x_it, f_t, Gamma_true, c_it

# Step 2: Run IPCA for different sample sizes and calculate Gamma error
sample_sizes = [10, 100, 200, 400, 600, 800, 1000]
errors = []

for N in sample_sizes:
    X, f_t, Gamma_true, c_it = simulate_data(N, T, K, L, Gamma_true, sigma_e)
    
    # Reshape data into panel form (N individuals, T time periods)
    X_flat = X.flatten()
    
    # Reshape c_it appropriately
    c_it_flat = c_it.reshape(N * T, L)  # Instruments reshape to (N*T, L)

    # Prepare dependent variables for IPCA
    y_flat = X_flat  # Dependent variable is already flattened

    # Step 2.1: Create entity-time MultiIndex for IPCA
    entities = np.repeat(np.arange(N), T)  # Entity index
    times = np.tile(np.arange(T), N)  # Time index

    index = pd.MultiIndex.from_arrays([entities, times], names=['entity', 'time'])

    # Convert c_it_flat and y_flat into DataFrames with MultiIndex
    c_it_df = pd.DataFrame(c_it_flat, index=index)
    y_df = pd.Series(y_flat, index=index)

    # Step 3: Fit IPCA model with Adjustments
    regr = InstrumentedPCA(n_factors=K, intercept=False, alpha=alpha_reg, max_iter=max_iterations, iter_tol=tolerance, n_jobs=-1)
    regr = regr.fit(X=c_it_df, y=y_df)

    # Retrieve the estimated Gamma (factor loadings)
    Gamma_est, _ = regr.get_factors(label_ind=True)

    # Compute squared error between true Gamma and estimated Gamma
    error = np.linalg.norm(Gamma_true - Gamma_est, ord='fro')**2
    errors.append(error)

# Step 4: Plot ||Gamma - Gamma*||^2 vs log(N)
fig, ax1 = plt.subplots()

# Primary axis: ||Gamma - Gamma*||^2 vs log(N)
ax1.plot(np.log(sample_sizes), errors, marker='o')
ax1.set_xlabel('log(N)')
ax1.set_ylabel(r'$\| \Gamma - \Gamma^* \|^2$')
ax1.set_title(r'$\| \Gamma - \Gamma^* \|^2$ vs Log(N)')
ax1.grid(True)

# Secondary axis: actual N values
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())  # Align the two x-axes
ax2.set_xticks(np.log(sample_sizes))  # Same locations as log(N)
ax2.set_xticklabels(sample_sizes)  # Label them with actual N values
ax2.set_xlabel('N (sample size)')

plt.show()

# Step 5: Plot log(||Gamma - Gamma*||^2) vs log(N)
log_errors = np.log(errors)

fig, ax1 = plt.subplots()

# Primary axis: log(||Gamma - Gamma*||^2) vs log(N)
ax1.plot(np.log(sample_sizes), log_errors, marker='o', color='red')
ax1.set_xlabel('log(N)')
ax1.set_ylabel(r'$\log(\| \Gamma - \Gamma^* \|^2)$', color='red')
ax1.set_title(r'$\log(\| \Gamma - \Gamma^* \|^2)$ vs Log(N)')
ax1.grid(True)

# Secondary axis: actual N values
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())  # Align the two x-axes
ax2.set_xticks(np.log(sample_sizes))  # Same locations as log(N)
ax2.set_xticklabels(sample_sizes)  # Label them with actual N values
ax2.set_xlabel('N (sample size)')

plt.show()
