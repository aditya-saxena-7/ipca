import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipca import InstrumentedPCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
from datetime import datetime

# Create base plots directory if it doesn't exist
base_plots_dir = "plots"
if not os.path.exists(base_plots_dir):
    os.makedirs(base_plots_dir)

# Create timestamped subdirectory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plots_dir = os.path.join(base_plots_dir, timestamp)
os.makedirs(plots_dir)
print(f"\nCreated directory for this run: {plots_dir}")

# Function to get filename (now without timestamp since it's in directory name)
def get_filename(base_name):
    return f"{base_name}.png"

# Simulation Parameters
T = 200  # Number of time periods
K = 3    # Number of latent factors
L = 5    # Number of observable instruments
alpha_reg = 0  # Increased regularization to avoid over-shrinking
max_iterations = 1000  # Maximum number of iterations
tolerance = 1e-3  # Convergence tolerance
sample_sizes = [10, 40, 70, 100, 130, 160, 190]
sigma_squares = [0.01, 0.03, 0.05]
num_simulations = 10  # Number of runs for averaging

# Save simulation parameters to a text file in the timestamped directory
params_file = os.path.join(plots_dir, "simulation_parameters.txt")
with open(params_file, 'w') as f:
    f.write(f"Simulation Parameters:\n")
    f.write(f"Time periods (T): {T}\n")
    f.write(f"Number of latent factors (K): {K}\n")
    f.write(f"Number of observable instruments (L): {L}\n")
    f.write(f"Alpha regularization: {alpha_reg}\n")
    f.write(f"Max iterations: {max_iterations}\n")
    f.write(f"Tolerance: {tolerance}\n")
    f.write(f"Sample sizes: {sample_sizes}\n")
    f.write(f"Sigma squares: {sigma_squares}\n")
    f.write(f"Number of simulations for averaging: {num_simulations}\n")
    f.write(f"\nTimestamp: {timestamp}")

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
    B = np.array([[0.4, 0.25, 0.15, 0.1, 0.1],
                [0.2, 0.5, 0.2, 0.05, 0.05],
                [0.15, 0.25, 0.4, 0.1, 0.1],
                [0.1, 0.2, 0.1, 0.45, 0.15],
                [0.1, 0.1, 0.15, 0.1, 0.45]])
    
    c_it = np.zeros((N, T, L))
    for i in range(N):
        for t in range(1, T):
            eps_it = np.random.normal(0, 0.1, L)
            c_it[i, t] = B @ c_it[i, t-1] + eps_it

    # Generate errors and observed data
    e_it = np.random.normal(0, sigma_e, (N, T))  # Error term
    x_it = np.zeros((N, T))
    for i in range(N):
        for t in range(T):
            x_it[i, t] = c_it[i, t] @ Gamma_true @ f_t[t] + e_it[i, t]

    return x_it, f_t, c_it

# Initialize true Gamma (factor loadings matrix)
Gamma_true = np.random.normal(0, 0.1, (L, K))

# Initialize StandardScaler for feature scaling
scaler = StandardScaler()

# Store average errors for different combinations of sigma^2 and N
errors = []

for sigma_e in sigma_squares:
    for N in sample_sizes:
        avg_error = 0
        for _ in range(num_simulations):
            # Simulate data
            X, f_t, c_it = simulate_data(N, T, K, L, Gamma_true, sigma_e)
            
            X_flat = X.flatten()
            c_it_flat = c_it.reshape(N * T, L)
            c_it_flat_scaled = scaler.fit_transform(c_it_flat)

            y_flat = X_flat
            entities = np.repeat(np.arange(N), T)
            times = np.tile(np.arange(T), N)
            index = pd.MultiIndex.from_arrays([entities, times], names=['entity', 'time'])
            
            c_it_df_scaled = pd.DataFrame(c_it_flat_scaled, index=index)
            y_df = pd.Series(y_flat, index=index)
            
            regr = InstrumentedPCA(n_factors=K, intercept=False, alpha=alpha_reg, 
                                 max_iter=max_iterations, l1_ratio=0.001, 
                                 iter_tol=tolerance, n_jobs=-1)
            try:
                regr = regr.fit(X=c_it_df_scaled, y=y_df, data_type="panel")
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Error encountered for N={N}, sigma^2={sigma_e}: {e}")
                continue

            Gamma_est, _ = regr.get_factors(label_ind=True)
            error = np.linalg.norm(Gamma_true - Gamma_est, ord='fro')**2
            avg_error += error

        avg_error /= num_simulations  # Average the error over simulations
        errors.append([sigma_e, N, avg_error])

# Convert errors to DataFrame and save to CSV
df_errors = pd.DataFrame(errors, columns=['sigma^2', 'N', 'Avg_Error'])
df_errors.to_csv(os.path.join(plots_dir, 'avg_errors.csv'), index=False)

# Plotting the heatmap of averaged errors
plt.figure(figsize=(10, 8))
pivot_table = df_errors.pivot(index="sigma^2", columns="N", values="Avg_Error")

if not pivot_table.empty:
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis")
    plt.title(r'Average $\| \Gamma - \Gamma^* \|^2$ for different $\sigma^2$ and N')
    plt.xlabel('Sample Size (N)')
    plt.ylabel(r'$\sigma^2$ (Noise Variance)')
    plt.figtext(0.99, 0.01, f"T={T}, K={K}, L={L}, max_iterations={max_iterations}, alpha_reg={alpha_reg}", horizontalalignment='right')
    
    heatmap_filename = get_filename("heatmap_avg")
    plt.savefig(os.path.join(plots_dir, heatmap_filename), bbox_inches='tight', dpi=300)
    plt.close()

# Plotting log-log of averaged errors vs sample size
plt.figure(figsize=(8, 6))
log_sample_sizes = np.log(df_errors['N'])
log_avg_errors = np.log(df_errors['Avg_Error'])
plt.plot(log_sample_sizes, log_avg_errors, marker='o')
plt.xlabel(r'log(N)')
plt.ylabel(r'log(Average $\| \Gamma - \Gamma^* \|^2$)')
plt.title(r'log(Average $\| \Gamma - \Gamma^* \|^2$) vs log(N)')
plt.grid(True)

log_plot_filename = get_filename("log_plot_avg")
plt.savefig(os.path.join(plots_dir, log_plot_filename), bbox_inches='tight', dpi=300)
plt.close()

# Modified log-log plot by sigma squares
plt.figure(figsize=(8, 6))

for sigma_e in sigma_squares:
    sigma_df = df_errors[df_errors['sigma^2'] == sigma_e]
    log_sample_sizes = np.log(sigma_df['N'])
    log_avg_errors = np.log(sigma_df['Avg_Error'])
    plt.plot(log_sample_sizes, log_avg_errors, marker='o', label=f'sigma^2={sigma_e}')

plt.xlabel(r'log(N)')
plt.ylabel(r'log(Average $\| \Gamma - \Gamma^* \|^2$)')
plt.title(r'log(Average $\| \Gamma - \Gamma^* \|^2$) vs log(N) by $\sigma^2$')
plt.legend(title=r'$\sigma^2$ (Noise Variance)')
plt.grid(True)

sigma_plot_filename = get_filename("sigma_comparison_avg")
plt.savefig(os.path.join(plots_dir, sigma_plot_filename), bbox_inches='tight', dpi=300)
plt.close()

print(f"\nAll results have been saved in: {plots_dir}")
print("Files saved:")
print(f"1. Simulation parameters: simulation_parameters.txt")
print(f"2. Average error data: avg_errors.csv")
print(f"3. Averaged Heatmap: {heatmap_filename}")
print(f"4. Averaged Log Plot: {log_plot_filename}")
print(f"5. Averaged Sigma Comparison: {sigma_plot_filename}")
