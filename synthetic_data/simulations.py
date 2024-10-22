import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Simulation Parameters
N = 200  # Number of individuals (stocks, assets, etc.)
T = 200  # Number of time periods
K = 3    # Number of latent factors
L = 5    # Number of observable instruments
sigma_e = 0.2  # Standard deviation of the error term

# Step 1: Generate latent factors (f_t)
# Factors follow a VAR(1) process: f_t = A * f_{t-1} + eta_t
A = np.array([[0.8, 0.1, 0.05],
              [0.1, 0.85, 0.05],
              [0.05, 0.1, 0.9]])  # Transition matrix
f_t = np.zeros((T, K))  # Initialize factors matrix

# Generate the factor time series
for t in range(1, T):
    eta_t = np.random.normal(0, 0.1, K)  # Noise term
    f_t[t] = A @ f_t[t-1] + eta_t

# Step 2: Generate observable instruments (c_{i,t})
# Instruments also follow a VAR(1) process: c_{i,t} = B * c_{i,t-1} + eps_{i,t}
B = np.array([[0.7, 0.1, 0.05, 0.05, 0.1],
              [0.05, 0.8, 0.1, 0.05, 0.05],
              [0.05, 0.05, 0.75, 0.1, 0.05],
              [0.1, 0.05, 0.1, 0.85, 0.05],
              [0.05, 0.05, 0.1, 0.05, 0.8]])  # Transition matrix for instruments
c_it = np.zeros((N, T, L))  # Instruments matrix

# Generate instruments for each individual
for i in range(N):
    for t in range(1, T):
        eps_it = np.random.normal(0, 0.1, L)  # Noise term for instruments
        c_it[i, t] = B @ c_it[i, t-1] + eps_it

# Step 3: Generate errors (e_{i,t})
e_it = np.random.normal(0, sigma_e, (N, T))  # Error term

# Step 4: Generate factor loadings (Gamma)
Gamma = np.random.normal(0, 0.1, (L, K))  # Loadings matrix (L instruments to K factors)

# Step 5: Generate observed data (x_{i,t})
# x_{i,t} = c_{i,t} * Gamma * f_t + e_{i,t}
x_it = np.zeros((N, T))

for i in range(N):
    for t in range(T):
        x_it[i, t] = c_it[i, t] @ Gamma @ f_t[t] + e_it[i, t]

# Visualize the latent factors over time
plt.plot(f_t)
plt.title("Simulated Latent Factors Over Time")
plt.xlabel("Time")
plt.ylabel("Factor Value")
plt.legend([f"Factor {i+1}" for i in range(K)])
plt.show()

# Show first few rows of the simulated data (x_{i,t})
print("First 5 rows of the simulated observed data (x_{i,t}):")
print(x_it[:5, :5])
