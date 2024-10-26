import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipca import InstrumentedPCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Simulation Parameters
T = 200  # Number of time periods
K = 3    # Number of latent factors
L = 5    # Number of observable instruments
alpha_reg = 0  # Increased regularization to avoid over-shrinking
max_iterations = 1000  # Maximum number of iterations
tolerance = 1e-3  # Convergence tolerance
sample_sizes = [30, 50, 70] #[10,15,20,25,35,40,45,50,55,60,65,70,75,80,85,90,95,100]  # Avoid small sample sizes to prevent matrix issues
sigma_squares = [0.01, 0.03] #[0.05, 0.07, 0.09, 0.1]  # Only use 0.05 for sigma^2

# Function to simulate data with time periods starting from 101
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
    # Further diversified transition matrix with more interaction and randomness
    B = np.array([[0.4, 0.25, 0.15, 0.1, 0.1],
                [0.2, 0.5, 0.2, 0.05, 0.05],
                [0.15, 0.25, 0.4, 0.1, 0.1],
                [0.1, 0.2, 0.1, 0.45, 0.15],
                [0.1, 0.1, 0.15, 0.1, 0.45]])  # Further diversified transition matrix
    
    c_it = np.zeros((N, T, L))
    for i in range(N):
        for t in range(1, T):
            #eps_it = np.random.normal(0, 0.1, L)  # Noise term for instruments
            eps_it = np.random.normal(0, 0.1, L)
            c_it[i, t] = B @ c_it[i, t-1] + eps_it

    # Generate errors and observed data
    e_it = np.random.normal(0, sigma_e, (N, T))  # Error term
    x_it = np.zeros((N, T))
    for i in range(N):
        for t in range(T):
            x_it[i, t] = c_it[i, t] @ Gamma_true @ f_t[t] + e_it[i, t]

    return x_it, f_t, c_it

def calculate_vif(c_it_flat_scaled):
    """Calculate VIF for each instrument."""
    vif_data = pd.DataFrame()
    vif_data["feature"] = [f"Instrument_{i}" for i in range(c_it_flat_scaled.shape[1])]
    vif_data["VIF"] = [variance_inflation_factor(c_it_flat_scaled, i) for i in range(c_it_flat_scaled.shape[1])]
    return vif_data

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

        # Calculate VIF for the standardized instruments
        vif_data = calculate_vif(c_it_flat_scaled)
        
        # Display VIF
        print(f"VIF for N={N}, sigma^2={sigma_e}")
        print(vif_data)
        print("\n")

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
            regr = regr.fit(X=c_it_df_scaled, y=y_df, data_type="panel")
        except np.linalg.LinAlgError as e:
            print(f"LinAlgError encountered for N={N}, sigma^2={sigma_e}: {e}")
            continue  # Skip this iteration
        except ValueError as e:  # This handles convergence issues like matrix problems
            print(f"ValueError encountered for N={N}, sigma^2={sigma_e}: {e}")
            continue  # Skip problematic configurations

        # Retrieve the estimated Gamma (factor loadings)
        Gamma_est, _ = regr.get_factors(label_ind=True)
        
        # Calculate the error ||Gamma - Gamma*||^2
        error = np.linalg.norm(Gamma_true - Gamma_est, ord='fro')**2
        
        # Store error with corresponding sigma^2 and N
        errors.append([sigma_e, N, error])

# Convert errors to DataFrame for easier plotting
df_errors = pd.DataFrame(errors, columns=['sigma^2', 'N', 'Error'])

# Plotting the heatmap
plt.figure(figsize=(10, 8))
pivot_table = df_errors.pivot(index="sigma^2", columns="N", values="Error")

if not pivot_table.empty:
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis")
    plt.title(r'$\| \Gamma - \Gamma^* \|^2$ for different $\sigma^2$ and N')
    plt.xlabel('Sample Size (N)')
    plt.ylabel(r'$\sigma^2$ (Noise Variance)')
    plt.figtext(0.99, 0.01, f"T={T}, K={K}, L={L}, sigma_e={sigma_e}, max_iterations={max_iterations}, alpha_reg={alpha_reg}", horizontalalignment='right')
    plt.show()
else:
    print("Heatmap data is empty, skipping heatmap plot.")

# Plotting log(||Gamma - Gamma*||^2) vs log(N)
log_errors = np.log([error[2] for error in errors])  # Extract log of errors
log_sample_sizes = np.log(sample_sizes)  # Log of sample sizes

valid_sample_sizes = []
valid_errors = []

for error in errors:
    if len(error) == 3:  # Ensure valid error values exist
        valid_sample_sizes.append(error[1])  # Append valid N
        valid_errors.append(error[2])  # Append valid error

# Ensure matching lengths for the plot
if len(valid_sample_sizes) == len(valid_errors):
    log_errors = np.log(valid_errors)
    log_sample_sizes = np.log(valid_sample_sizes)

    # Plotting log(||Gamma - Gamma*||^2) vs log(N)
    plt.figure(figsize=(8, 6))
    plt.plot(log_sample_sizes, log_errors, marker='o')
    plt.xlabel(r'log(N)')
    plt.ylabel(r'log($\| \Gamma - \Gamma^* \|^2$)')
    plt.title(r'log($\| \Gamma - \Gamma^* \|^2$) vs log(N)')
    plt.grid(True)
    plt.show()
else:
    print("Mismatch in lengths of sample sizes and errors.")

# Modified plotting for log(||Gamma - Gamma*||^2) vs log(N) with different colors for sigma_squares
plt.figure(figsize=(8, 6))

# Loop through each sigma^2 value
for sigma_e in sigma_squares:
    # Filter the data for the current sigma^2
    sigma_df = df_errors[df_errors['sigma^2'] == sigma_e]
    
    # Extract log of sample sizes and errors
    valid_sample_sizes = sigma_df['N']
    valid_errors = sigma_df['Error']
    
    # Skip any missing values
    valid_sample_sizes = valid_sample_sizes[~valid_errors.isna()]
    valid_errors = valid_errors.dropna()
    
    # Ensure matching lengths for the plot
    if len(valid_sample_sizes) == len(valid_errors) and len(valid_sample_sizes) > 0:
        log_sample_sizes = np.log(valid_sample_sizes)
        log_errors = np.log(valid_errors)
        
        # Plot the data with different color for each sigma^2
        plt.plot(log_sample_sizes, log_errors, marker='o', label=f'sigma^2={sigma_e}')
    
# Add labels and title to the plot
plt.xlabel(r'log(N)')
plt.ylabel(r'log($\| \Gamma - \Gamma^* \|^2$)')
plt.title(r'log($\| \Gamma - \Gamma^* \|^2$) vs log(N)')
plt.legend(title=r'$\sigma^2$ (Noise Variance)')
plt.grid(True)
plt.show()



