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

# [Rest of the simulation parameters and functions remain the same until plotting code]

# Simulation Parameters
T = 200  # Number of time periods
K = 3    # Number of latent factors
L = 5    # Number of observable instruments
alpha_reg = 0  # Increased regularization to avoid over-shrinking
max_iterations = 1000  # Maximum number of iterations
tolerance = 1e-3  # Convergence tolerance
sample_sizes = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200] #[30, 50, 70]
sigma_squares = [0.01,0.03,0.05,0.07,0.09,0.1] #[0.01, 0.03] 

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

# Also save VIF data to a file
vif_file = os.path.join(plots_dir, "vif_results.txt")
vif_output = open(vif_file, 'w')

for sigma_e in sigma_squares:
    for N in sample_sizes:
        # Simulate data
        X, f_t, c_it = simulate_data(N, T, K, L, Gamma_true, sigma_e)
        
        X_flat = X.flatten()
        c_it_flat = c_it.reshape(N * T, L)
        c_it_flat_scaled = scaler.fit_transform(c_it_flat)

        # Calculate and save VIF
        vif_data = calculate_vif(c_it_flat_scaled)
        vif_output.write(f"\nVIF for N={N}, sigma^2={sigma_e}\n")
        vif_output.write(str(vif_data))
        vif_output.write("\n" + "="*50 + "\n")
        
        print(f"VIF for N={N}, sigma^2={sigma_e}")
        print(vif_data)
        print("\n")

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
        except np.linalg.LinAlgError as e:
            print(f"LinAlgError encountered for N={N}, sigma^2={sigma_e}: {e}")
            continue
        except ValueError as e:
            print(f"ValueError encountered for N={N}, sigma^2={sigma_e}: {e}")
            continue

        Gamma_est, _ = regr.get_factors(label_ind=True)
        error = np.linalg.norm(Gamma_true - Gamma_est, ord='fro')**2
        errors.append([sigma_e, N, error])

vif_output.close()

# Convert errors to DataFrame and save to CSV
df_errors = pd.DataFrame(errors, columns=['sigma^2', 'N', 'Error'])
df_errors.to_csv(os.path.join(plots_dir, 'errors.csv'), index=False)

# Plotting the heatmap
plt.figure(figsize=(10, 8))
pivot_table = df_errors.pivot(index="sigma^2", columns="N", values="Error")

if not pivot_table.empty:
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis")
    plt.title(r'$\| \Gamma - \Gamma^* \|^2$ for different $\sigma^2$ and N')
    plt.xlabel('Sample Size (N)')
    plt.ylabel(r'$\sigma^2$ (Noise Variance)')
    plt.figtext(0.99, 0.01, f"T={T}, K={K}, L={L}, sigma_e={sigma_e}, max_iterations={max_iterations}, alpha_reg={alpha_reg}", horizontalalignment='right')
    
    heatmap_filename = get_filename("heatmap")
    plt.savefig(os.path.join(plots_dir, heatmap_filename), bbox_inches='tight', dpi=300)
    plt.close()
else:
    print("Heatmap data is empty, skipping heatmap plot.")

# Plotting log(||Gamma - Gamma*||^2) vs log(N)
valid_sample_sizes = []
valid_errors = []

for error in errors:
    if len(error) == 3:
        valid_sample_sizes.append(error[1])
        valid_errors.append(error[2])

if len(valid_sample_sizes) == len(valid_errors):
    log_errors = np.log(valid_errors)
    log_sample_sizes = np.log(valid_sample_sizes)

    plt.figure(figsize=(8, 6))
    plt.plot(log_sample_sizes, log_errors, marker='o')
    plt.xlabel(r'log(N)')
    plt.ylabel(r'log($\| \Gamma - \Gamma^* \|^2$)')
    plt.title(r'log($\| \Gamma - \Gamma^* \|^2$) vs log(N)')
    plt.grid(True)
    
    log_plot_filename = get_filename("log_plot")
    plt.savefig(os.path.join(plots_dir, log_plot_filename), bbox_inches='tight', dpi=300)
    plt.close()
else:
    print("Mismatch in lengths of sample sizes and errors.")

# Modified plotting for different sigma_squares
plt.figure(figsize=(8, 6))

for sigma_e in sigma_squares:
    sigma_df = df_errors[df_errors['sigma^2'] == sigma_e]
    valid_sample_sizes = sigma_df['N']
    valid_errors = sigma_df['Error']
    
    valid_sample_sizes = valid_sample_sizes[~valid_errors.isna()]
    valid_errors = valid_errors.dropna()
    
    if len(valid_sample_sizes) == len(valid_errors) and len(valid_errors) > 0:
        log_sample_sizes = np.log(valid_sample_sizes)
        log_errors = np.log(valid_errors)
        plt.plot(log_sample_sizes, log_errors, marker='o', label=f'sigma^2={sigma_e}')

plt.xlabel(r'log(N)')
plt.ylabel(r'log($\| \Gamma - \Gamma^* \|^2$)')
plt.title(r'log($\| \Gamma - \Gamma^* \|^2$) vs log(N)')
plt.legend(title=r'$\sigma^2$ (Noise Variance)')
plt.grid(True)

sigma_plot_filename = get_filename("sigma_comparison")
plt.savefig(os.path.join(plots_dir, sigma_plot_filename), bbox_inches='tight', dpi=300)
plt.close()

print(f"\nAll results have been saved in: {plots_dir}")
print("Files saved:")
print(f"1. Simulation parameters: simulation_parameters.txt")
print(f"2. VIF results: vif_results.txt")
print(f"3. Error data: errors.csv")
print(f"4. Heatmap: {heatmap_filename}")
print(f"5. Log Plot: {log_plot_filename}")
print(f"6. Sigma Comparison: {sigma_plot_filename}")