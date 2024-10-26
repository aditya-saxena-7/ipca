### Alignment Verification and Error Analysis

To compare the estimated $\Gamma$ and true $\Gamma^*$ values and analyze alignment errors, follow these steps:

1. **Calculate Alignment Error**: Compute $\| \Gamma - \Gamma^* \|^2$ for each configuration of $N$ and $\sigma^2$. This is already done, but to analyze trends, aggregate these errors across noise levels or sample sizes separately.

2. **Rotational Ambiguity Check**: Since factor models can be rotationally invariant, the estimated $\Gamma$ might be rotated compared to $\Gamma^*$. This can cause misalignment, making it challenging to interpret the difference as an actual error. Procrustes Analysis can address this.

---

### Implementing Procrustes Analysis for Alignment Correction

**Procrustes Analysis** aligns two matrices (e.g., $\Gamma$ and $\Gamma^*$) by finding the best rotation, scaling, and reflection that minimizes the discrepancy between them. This will help to verify if the differences observed in $|\Gamma - \Gamma^*|$ are due to rotational ambiguity or genuine estimation error.

#### Steps for Procrustes Analysis

1. **Standardize**: Ensure both $\Gamma$ and $\Gamma^*$ have zero mean. Centering can help remove any translation differences.

2. **Apply Procrustes Transformation**:
   - Calculate the optimal rotation matrix $R$ that aligns $\Gamma$ to $\Gamma^*$ by solving the Procrustes problem:

     $$
     \min_{R} \| \Gamma R - \Gamma^* \|_F
     $$

     where $\| \cdot \|_F$ denotes the Frobenius norm, and $R$ is an orthogonal matrix (i.e., $R^T R = I$).
   - This can be done using singular value decomposition (SVD) on $\Gamma^{*T} \Gamma$.

3. **Transform**: Rotate the estimated $\Gamma$ using $R$ to get $\Gamma_{\text{aligned}} = \Gamma R$.

4. **Evaluate Alignment**: Calculate $\| \Gamma_{\text{aligned}} - \Gamma^* \|$. If this aligned error is significantly lower than the unaligned error, it suggests that rotational ambiguity was impacting the results. Otherwise, the observed error likely stems from model limitations or noise effects.

By implementing these steps, you should be able to determine whether the discrepancies in $|\Gamma - \Gamma^*|$ are due to alignment issues or if they reflect limitations of the IPCA model under the given conditions.

---

### Implementation 

Here’s an explanation of the code [simulations_6_4.py](https://github.com/aditya-saxena-7/ipca/blob/main/synthetic_data/simulations_6_4.py):

### 2. **Defining Simulation Parameters**

```python
# Simulation Parameters
T = 200  # Number of time periods
K = 3    # Number of latent factors
L = 5    # Number of observable instruments
alpha_reg = 0  # Increased regularization to avoid over-shrinking
max_iterations = 1000  # Maximum number of iterations
tolerance = 1e-3  # Convergence tolerance
sample_sizes = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
sigma_squares = [0.01,0.03,0.05,0.07,0.09,0.1]
num_simulations = 1  # Number of runs for averaging
```

This block defines the simulation parameters:
- **T**: Number of time periods.
- **K**: Number of latent factors.
- **L**: Number of observable instruments.
- **alpha_reg**: Regularization parameter for IPCA.
- **max_iterations** and **tolerance**: Controls for IPCA convergence.
- **sample_sizes** and **sigma_squares**: Lists of different sample sizes and noise variances to iterate over.
- **num_simulations**: Number of times each configuration is run to compute average errors.

---

### 3. **Data Simulation Function**

```python
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
```

The `simulate_data()` function generates synthetic data for each simulation:
- **Latent Factors \( f_t \)**: Generated with a transition matrix \( A \) to introduce temporal dependency.
- **Instrument Variables \( c_{it} \)**: Created using matrix \( B \) with some noise.
- **Observed Data \( x_{it} \)**: Computed as a function of \( c_{it} \), \( \Gamma_{\text{true}} \), \( f_t \), and an error term \( e_{it} \).

---

### 4. Error Calculation and Procrustes Alignment

#### Overview
This section of the code is responsible for:
1. Calculating the **unaligned error** between the true and estimated factor loadings.
2. Applying **Procrustes Analysis** to align the estimated factor loadings to the true factor loadings, which corrects for rotational ambiguities.
3. Calculating the **aligned error** after Procrustes alignment.
4. Averaging these errors across multiple runs to assess the overall impact of Procrustes alignment on reducing error.

---

#### Code Breakdown

##### Step 1: Looping Over Noise and Sample Size Configurations

```python
for sigma_e in sigma_squares:
    for N in sample_sizes:
        avg_error = 0
        avg_aligned_error = 0
```

The code iterates over each combination of noise variance (`sigma_e`) and sample size (`N`). For each configuration:
- `avg_error` and `avg_aligned_error` are initialized to zero. These variables will store the cumulative unaligned and aligned errors across multiple simulations for averaging.

---

##### Step 2: Running Simulations and Fitting the IPCA Model

```python
        for _ in range(num_simulations):
            # Simulate data
            X, f_t, c_it = simulate_data(N, T, K, L, Gamma_true, sigma_e)
```

Within each configuration, we run the specified number of simulations (`num_simulations`) to calculate averaged errors:
1. **Data Simulation**: `simulate_data()` generates synthetic data for the current configuration of `N` and `sigma_e`.

---

##### Step 3: Preparing Data for IPCA

```python
            X_flat = X.flatten()
            c_it_flat = c_it.reshape(N * T, L)
            c_it_flat_scaled = scaler.fit_transform(c_it_flat)

            y_flat = X_flat
            entities = np.repeat(np.arange(N), T)
            times = np.tile(np.arange(T), N)
            index = pd.MultiIndex.from_arrays([entities, times], names=['entity', 'time'])
            
            c_it_df_scaled = pd.DataFrame(c_it_flat_scaled, index=index)
            y_df = pd.Series(y_flat, index=index)
```

This block reshapes and scales the simulated data for IPCA:
- `X_flat`: Flattens the observed data for compatibility with IPCA.
- `c_it_flat_scaled`: Reshapes and scales the instruments data.
- `y_flat`: Flattens the observed output data.
- `c_it_df_scaled` and `y_df`: Organize data as a panel data structure with entity-time indices for use with IPCA.

---

##### Step 4: Fitting the IPCA Model

```python
            regr = InstrumentedPCA(n_factors=K, intercept=False, alpha=alpha_reg, 
                                 max_iter=max_iterations, l1_ratio=0.001, 
                                 iter_tol=tolerance, n_jobs=-1)
            try:
                regr = regr.fit(X=c_it_df_scaled, y=y_df, data_type="panel")
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Error encountered for N={N}, sigma^2={sigma_e}: {e}")
                continue
```

The IPCA model is configured and fitted to the data:
- **n_factors**: Specifies the number of latent factors.
- **alpha**: Regularization parameter.
- **max_iter** and **iter_tol**: Set convergence controls for the model.
  
If fitting the model fails (e.g., due to singular matrices), the code catches the error and continues to the next simulation.

---

##### Step 5: Calculating the Unaligned Error

```python
            Gamma_est, _ = regr.get_factors(label_ind=True)
            # Calculate unaligned error
            unaligned_error = np.linalg.norm(Gamma_true - Gamma_est, ord='fro')**2
            avg_error += unaligned_error
```

After fitting the model, the estimated factor loadings matrix, `Gamma_est`, is obtained. 
- **Unaligned Error Calculation**: The unaligned error is computed as the Frobenius norm squared:
  \[
  \text{Unaligned Error} = \| \Gamma_{\text{true}} - \Gamma_{\text{est}} \|_F^2
  \]
  This represents the discrepancy between the true and estimated factor loadings before any alignment correction. The unaligned error is then added to `avg_error` to calculate an average across multiple simulations.

---

##### Step 6: Applying Procrustes Analysis for Alignment

```python
            # Apply Procrustes Analysis for alignment
            _, aligned_Gamma_est, disparity = procrustes(Gamma_true, Gamma_est)
            aligned_error = np.linalg.norm(Gamma_true - aligned_Gamma_est, ord='fro')**2
            avg_aligned_error += aligned_error
```

To address rotational ambiguity, **Procrustes Analysis** aligns `Gamma_est` with `Gamma_true`:
1. **Procrustes Alignment**: The function `procrustes(Gamma_true, Gamma_est)` calculates a transformation to best align `Gamma_est` with `Gamma_true` in terms of rotation, scaling, and translation.
   - `aligned_Gamma_est` is the aligned version of `Gamma_est` after Procrustes Analysis.
   - `disparity` is a measure of the alignment fit (not used in this calculation).
   
2. **Aligned Error Calculation**: After alignment, the aligned error is calculated similarly:
   \[
   \text{Aligned Error} = \| \Gamma_{\text{true}} - \Gamma_{\text{aligned}} \|_F^2
   \]
   This error represents the residual discrepancy after correcting for rotational alignment. The aligned error is added to `avg_aligned_error` for averaging over simulations.

---

##### Step 7: Averaging the Errors

```python
        avg_error /= num_simulations  # Average unaligned error
        avg_aligned_error /= num_simulations  # Average aligned error
        errors.append([sigma_e, N, avg_error, avg_aligned_error])
```

Once all simulations are completed for a particular configuration of `sigma_e` and `N`, the cumulative errors are averaged:
- `avg_error`: Average unaligned error across simulations.
- `avg_aligned_error`: Average aligned error across simulations.

The results for this configuration are stored in the `errors` list.

---

### Significance of Procrustes Alignment and Error Calculation

#### Purpose of Procrustes Alignment
In factor analysis, **rotational ambiguity** can cause the estimated factor loadings to be misaligned with the true loadings even if they capture the correct underlying structure. Procrustes Analysis:
- Corrects for these ambiguities by applying optimal rotation, scaling, and translation to align `Gamma_est` with `Gamma_true`.
- Enables a more accurate assessment of model performance by ensuring that the error calculation only reflects discrepancies beyond mere rotations.

#### Significance of Error Types
- **Unaligned Error**: Reflects the discrepancy between `Gamma_true` and `Gamma_est` before alignment. High unaligned errors often indicate rotational misalignment, especially under high noise or small sample sizes.
- **Aligned Error**: Shows the discrepancy after Procrustes alignment, providing a clearer view of the model’s accuracy in capturing the underlying factors without the confounding effect of rotational ambiguity.

By comparing these errors across various noise levels and sample sizes, we can evaluate:
- The effectiveness of Procrustes alignment.
- How noise and sample size impact the model’s ability to estimate factor loadings accurately.
---

### 5. **Plotting Results**

Plots for the above code can be found with interpretation [here](https://github.com/aditya-saxena-7/ipca/tree/main/synthetic_data/plots)

All plots are dumped here: [20241026_133634](https://github.com/aditya-saxena-7/ipca/tree/main/synthetic_data/plots/20241026_133634)
