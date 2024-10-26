import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipca import InstrumentedPCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.spatial import procrustes
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
sample_sizes = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200] #[30, 50, 70]
sigma_squares = [0.01,0.03,0.05,0.07,0.09,0.1] #[0.01, 0.03] 
num_simulations = 1  # Number of runs for averaging

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

# Store errors for different combinations of sigma^2 and N
errors = []

for sigma_e in sigma_squares:
    for N in sample_sizes:
        avg_error = 0
        avg_aligned_error = 0
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
            
            # Calculate unaligned error
            unaligned_error = np.linalg.norm(Gamma_true - Gamma_est, ord='fro')**2
            avg_error += unaligned_error

            # Apply Procrustes Analysis for alignment
            _, aligned_Gamma_est, disparity = procrustes(Gamma_true, Gamma_est)
            aligned_error = np.linalg.norm(Gamma_true - aligned_Gamma_est, ord='fro')**2
            avg_aligned_error += aligned_error

        avg_error /= num_simulations  # Average unaligned error
        avg_aligned_error /= num_simulations  # Average aligned error
        errors.append([sigma_e, N, avg_error, avg_aligned_error])

# Convert errors to DataFrame and save to CSV
df_errors = pd.DataFrame(errors, columns=['sigma^2', 'N', 'Avg_Unaligned_Error', 'Avg_Aligned_Error'])
df_errors.to_csv(os.path.join(plots_dir, 'alignment_errors.csv'), index=False)

# Plotting heatmaps of unaligned and aligned errors
for error_type in ['Avg_Unaligned_Error', 'Avg_Aligned_Error']:
    plt.figure(figsize=(10, 8))
    pivot_table = df_errors.pivot(index="sigma^2", columns="N", values=error_type)
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis")
    plt.title(f'{error_type} for different $\sigma^2$ and N')
    plt.xlabel('Sample Size (N)')
    plt.ylabel(r'$\sigma^2$ (Noise Variance)')
    plt.figtext(0.99, 0.01, f"T={T}, K={K}, L={L}, max_iterations={max_iterations}, alpha_reg={alpha_reg}", horizontalalignment='right')
    
    heatmap_filename = get_filename(f"{error_type}_heatmap")
    plt.savefig(os.path.join(plots_dir, heatmap_filename), bbox_inches='tight', dpi=300)
    plt.close()

# Plotting log-log comparison of unaligned and aligned errors
plt.figure(figsize=(8, 6))
for error_type in ['Avg_Unaligned_Error', 'Avg_Aligned_Error']:
    log_sample_sizes = np.log(df_errors['N'])
    log_errors = np.log(df_errors[error_type])
    plt.plot(log_sample_sizes, log_errors, marker='o', label=error_type)

plt.xlabel(r'log(N)')
plt.ylabel(r'log(Error)')
plt.title(r'log(Unaligned vs. Aligned $\| \Gamma - \Gamma^* \|^2$) vs log(N)')
plt.legend()
plt.grid(True)

log_plot_filename = get_filename("log_alignment_comparison")
plt.savefig(os.path.join(plots_dir, log_plot_filename), bbox_inches='tight', dpi=300)
plt.close()

# Modified log-log plot for each sigma^2 with alignment comparison
for sigma_e in sigma_squares:
    plt.figure(figsize=(8, 6))
    sigma_df = df_errors[df_errors['sigma^2'] == sigma_e]
    for error_type in ['Avg_Unaligned_Error', 'Avg_Aligned_Error']:
        log_sample_sizes = np.log(sigma_df['N'])
        log_errors = np.log(sigma_df[error_type])
        plt.plot(log_sample_sizes, log_errors, marker='o', label=error_type)
    plt.xlabel(r'log(N)')
    plt.ylabel(r'log(Error)')
    plt.title(rf'log(Error) vs log(N) for $\sigma^2={sigma_e}$')
    plt.legend()
    plt.grid(True)

    sigma_plot_filename = get_filename(f"sigma_{sigma_e}_alignment_comparison")
    plt.savefig(os.path.join(plots_dir, sigma_plot_filename), bbox_inches='tight', dpi=300)
    plt.close()

print(f"\nAll results have been saved in: {plots_dir}")
print("Files saved:")
print(f"1. Simulation parameters: simulation_parameters.txt")
print(f"2. Alignment error data: alignment_errors.csv")
print(f"3. Heatmaps (Unaligned and Aligned Errors): {error_type}_heatmap")
print(f"4. Log Comparison Plot: {log_plot_filename}")
print(f"5. Sigma Comparison Plots: sigma_<sigma_e>_alignment_comparison")

# Additional Data Preparation for New Plots
df_errors['Error_Difference'] = df_errors['Avg_Unaligned_Error'] - df_errors['Avg_Aligned_Error']
df_errors['Percentage_Improvement'] = (df_errors['Error_Difference'] / df_errors['Avg_Unaligned_Error']) * 100

# 1. Plotting Error Difference (Unaligned - Aligned)
plt.figure(figsize=(10, 8))
pivot_table = df_errors.pivot(index="sigma^2", columns="N", values="Error_Difference")
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Difference in Error (Unaligned - Aligned) for different $\sigma^2$ and N')
plt.xlabel('Sample Size (N)')
plt.ylabel(r'$\sigma^2$ (Noise Variance)')
plt.figtext(0.99, 0.01, f"T={T}, K={K}, L={L}, max_iterations={max_iterations}, alpha_reg={alpha_reg}", horizontalalignment='right')

error_diff_filename = get_filename("error_difference_heatmap")
plt.savefig(os.path.join(plots_dir, error_diff_filename), bbox_inches='tight', dpi=300)
plt.close()

# 2. Plotting Percentage Improvement Due to Alignment
plt.figure(figsize=(10, 8))
pivot_table = df_errors.pivot(index="sigma^2", columns="N", values="Percentage_Improvement")
sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title('Percentage Improvement Due to Alignment for different $\sigma^2$ and N')
plt.xlabel('Sample Size (N)')
plt.ylabel(r'$\sigma^2$ (Noise Variance)')
plt.figtext(0.99, 0.01, f"T={T}, K={K}, L={L}, max_iterations={max_iterations}, alpha_reg={alpha_reg}", horizontalalignment='right')

perc_improv_filename = get_filename("percentage_improvement_heatmap")
plt.savefig(os.path.join(plots_dir, perc_improv_filename), bbox_inches='tight', dpi=300)
plt.close()

# 3. Plotting Individual Run Errors for Unaligned and Aligned
# For each combination of sigma^2 and N, we can record individual run errors for finer analysis
errors_across_runs = []

for sigma_e in sigma_squares:
    for N in sample_sizes:
        for run in range(num_simulations):
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
            
            # Calculate unaligned error
            unaligned_error = np.linalg.norm(Gamma_true - Gamma_est, ord='fro')**2

            # Apply Procrustes Analysis for alignment
            _, aligned_Gamma_est, disparity = procrustes(Gamma_true, Gamma_est)
            aligned_error = np.linalg.norm(Gamma_true - aligned_Gamma_est, ord='fro')**2

            # Store results
            errors_across_runs.append([sigma_e, N, run + 1, unaligned_error, aligned_error])

# Convert errors_across_runs to DataFrame and save
df_errors_across_runs = pd.DataFrame(errors_across_runs, columns=['sigma^2', 'N', 'Run', 'Unaligned_Error', 'Aligned_Error'])
df_errors_across_runs.to_csv(os.path.join(plots_dir, 'errors_across_runs.csv'), index=False)

# Plotting Unaligned and Aligned Errors across Runs for each combination of sigma^2 and N
for sigma_e in sigma_squares:
    for N in sample_sizes:
        subset = df_errors_across_runs[(df_errors_across_runs['sigma^2'] == sigma_e) & (df_errors_across_runs['N'] == N)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(subset['Run'], subset['Unaligned_Error'], marker='o', label='Unaligned Error')
        plt.plot(subset['Run'], subset['Aligned_Error'], marker='o', linestyle='--', label='Aligned Error')
        plt.title(f'Unaligned vs. Aligned Errors Across Runs (N={N}, $\sigma^2={sigma_e}$)')
        plt.xlabel('Run')
        plt.ylabel(r'$\| \Gamma - \Gamma^* \|^2$')
        plt.legend()
        plt.grid(True)
        
        run_error_filename = get_filename(f"error_across_runs_N_{N}_sigma_{sigma_e}")
        plt.savefig(os.path.join(plots_dir, run_error_filename), bbox_inches='tight', dpi=300)
        plt.close()

print(f"\nAdditional plots have been saved in: {plots_dir}")
print("Files saved:")
print(f"1. Error Difference Heatmap: {error_diff_filename}")
print(f"2. Percentage Improvement Heatmap: {perc_improv_filename}")
print(f"3. Individual Run Errors: errors_across_runs.csv")
print(f"4. Error Across Runs Plots: error_across_runs_N_<N>_sigma_<sigma_e>")
