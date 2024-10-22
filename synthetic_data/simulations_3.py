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


''' Simulation Parameters Description:
### 1. **T (Number of time periods)**
   - **Description**: This parameter represents the number of time periods over which the data is observed. In panel data, you observe multiple entities (like stocks, companies, or assets) over a set number of time periods. 
   - **In your context**: You have \( T = 200 \), meaning that for each entity (e.g., a stock or asset), the data is generated for 200 time periods.
   - **Why it's important**: The time series nature of the data allows the model to capture how the factors evolve over time and their effects on the dependent variables.

### 2. **K (Number of latent factors)**
   - **Description**: Latent factors are unobserved (hidden) variables that influence the observed data. The IPCA model tries to estimate these hidden factors based on the data.
   - **In your context**: \( K = 3 \), meaning the model assumes that there are 3 underlying latent factors influencing the data. 
   - **Why it's important**: These latent factors represent the core drivers of the dynamics in your system (e.g., market risk factors affecting asset returns). The goal of IPCA is to estimate these factors and their relationship to the observed data.

### 3. **L (Number of observable instruments)**
   - **Description**: Instruments are the observed variables used as predictors in the model. They are variables that are correlated with the latent factors but can be measured directly.
   - **In your context**: \( L = 5 \), meaning that there are 5 observable instruments for each entity (e.g., characteristics or features of the stocks, like book-to-market ratio, momentum, etc.).
   - **Why it's important**: These instruments are used to estimate the latent factors in the IPCA model. Their quality and number significantly influence the accuracy of the model's factor estimation.

### 4. **sigma_e (Standard deviation of the error term)**
   - **Description**: This represents the standard deviation of the noise (or error term) added to the data. In real-world data, there is always some random noise or uncertainty that can't be explained by the model.
   - **In your context**: \( \sigma_e = 0.2 \), meaning that the model assumes that the error (random noise) has a standard deviation of 0.2.
   - **Why it's important**: The error term represents randomness or unpredictable variations in the data. The larger the noise, the harder it is for the model to accurately estimate the latent factors. If the error is too large relative to the signal, the model's performance will suffer.

### 5. **alpha_reg (Regularization constant for Gamma estimation)**
   - **Description**: Regularization helps to prevent overfitting by penalizing large values in the model's parameters (in this case, the factor loadings \(\Gamma\)). This term controls the strength of that penalty.
   - **In your context**: \( \alpha_{\text{reg}} = 1 \times 10^{-4} \) introduces a small regularization term. A larger value would introduce more shrinkage, reducing the complexity of the estimated factor loadings.
   - **Why it's important**: Regularization is crucial when dealing with high-dimensional data or when there is a risk of overfitting, as it helps smooth out the model's parameter estimates and prevents the model from learning noise in the data.

### 6. **max_iterations (Maximum number of alternating least squares updates)**
   - **Description**: This sets the upper limit on how many times the IPCA model will iterate to try and find the best estimates for the latent factors and loadings.
   - **In your context**: \( \text{max_iterations} = 100 \), meaning that the IPCA model will perform at most 100 iterations of the alternating least squares (ALS) method.
   - **Why it's important**: The higher this value, the more time the model will spend trying to converge to an optimal solution. However, if the model is stuck or converging too slowly, it may take too long to compute. Limiting the iterations prevents it from running indefinitely but might lead to suboptimal solutions if set too low.

### 7. **tolerance (Convergence tolerance)**
   - **Description**: This is the threshold for stopping the iterative algorithm. If the change in the model's estimates between iterations is smaller than this tolerance, the algorithm will stop.
   - **In your context**: \( \text{tolerance} = 1 \times 10^{-4} \), meaning that the model will stop iterating once the updates between iterations become smaller than this value.
   - **Why it's important**: A smaller tolerance ensures that the model converges to a more precise solution but may take longer to reach that point. A larger tolerance speeds up convergence but may result in a less accurate solution.

### 8. **n_jobs (Number of parallel jobs)**
   - **Description**: This parameter controls how many CPU cores are used for parallel computation. 
   - **In your context**: \( n_{\text{jobs}} = -1 \), meaning that all available CPU cores will be used for parallelizing the computations. This speeds up the model fitting process.
   - **Why it's important**: For large datasets or complex models, parallel computation can significantly reduce the runtime of the algorithm. Setting \( n_{\text{jobs}} = -1 \) maximizes the computational resources.
'''

''' Step 1:

### What is \(\Gamma^*\)?

In the context of your model:
- \(\Gamma^*\) is the matrix of **true factor loadings**. This matrix defines how the latent factors \( f_t \) (which are hidden) influence the observable instruments (the features you can measure, like financial characteristics).
- It has dimensions \(L \times K\), where:
  - **L**: Number of observable instruments (i.e., characteristics or features of the entities like stock prices, firm size, etc.)
  - **K**: Number of latent factors (unobserved variables driving the system, such as macroeconomic factors or risk factors in finance).
- The latent factors \(f_t\) evolve over time, and the factor loadings matrix \(\Gamma^*\) represents the strength of the relationship between the observable instruments and the latent factors.

### Code Breakdown:
```python
Gamma_true = np.random.normal(0, 0.1, (L, K))  # True loadings matrix (L instruments to K factors)
```

Let's break this down step by step:

1. **\(\Gamma^*\) is generated randomly**:
   - The matrix \(\Gamma^*\) is generated using the `np.random.normal()` function, which draws random samples from a **normal distribution** (Gaussian distribution).
   - The arguments passed to `np.random.normal()` are:
     - **mean = 0**: The mean of the normal distribution from which the values are drawn is set to 0.
     - **standard deviation = 0.1**: The standard deviation (spread) of the normal distribution is set to 0.1, meaning the values will generally lie close to 0 but can vary slightly.
     - **(L, K)**: This specifies the shape of the matrix \(\Gamma^*\), where \(L\) is the number of observable instruments, and \(K\) is the number of latent factors.

   **Example**:
   If \(L = 5\) (5 instruments) and \(K = 3\) (3 latent factors), then \(\Gamma^*\) will be a \(5 \times 3\) matrix. This means:
   - Each row corresponds to one observable instrument.
   - Each column corresponds to a latent factor.
   - The entries of this matrix represent how strongly each instrument is related to each factor.

2. **Purpose of \(\Gamma^*\)**:
   - This matrix represents the **true relationship** between the observable instruments and the latent factors.
   - In the context of the IPCA model, the goal is to **estimate** this matrix based on the observed data. The estimated factor loadings are denoted as \(\Gamma\), and ideally, they should be close to the true matrix \(\Gamma^*\).
   - By generating \(\Gamma^*\) randomly, you're simulating a realistic scenario where the true underlying relationships are unknown but can be estimated based on data.

3. **Why normal distribution?**
   - The choice of normal distribution with mean 0 and standard deviation 0.1 allows for generating small values around 0, which ensures that the relationships between instruments and factors are not too large or unrealistic.
   - In practice, these relationships (factor loadings) often tend to be small, and this initialization reflects that. Large factor loadings might lead to unstable dynamics in the system, which is why moderate-sized values are typically preferred.

4. **Interpretation**:
   - Each element of \(\Gamma^*\) represents the **strength** of the relationship between a specific observable instrument (e.g., stock characteristic) and a latent factor (e.g., market factor).
   - For instance, the element in the first row and first column of \(\Gamma^*\) might represent how much the first observable instrument (say, stock price) is influenced by the first latent factor (e.g., market-wide risk).
   - The larger the value of an element in \(\Gamma^*\), the stronger the influence of the corresponding latent factor on the observable instrument.

### Example of \(\Gamma^*\):
If you have \(L = 5\) instruments and \(K = 3\) latent factors, \(\Gamma^*\) might look like this:

\[
\Gamma^* = 
\begin{bmatrix}
0.05 & -0.02 & 0.10 \\
-0.01 & 0.07 & -0.03 \\
0.04 & 0.02 & 0.08 \\
-0.06 & 0.01 & 0.09 \\
0.03 & -0.05 & 0.02
\end{bmatrix}
\]

In this matrix:
- The first row corresponds to the loadings of the first observable instrument with each of the three latent factors.
- For example, the first row values \(0.05, -0.02, 0.10\) indicate that the first instrument has a moderate positive relationship with factor 1, a small negative relationship with factor 2, and a strong positive relationship with factor 3.
- Similarly, each subsequent row shows the relationships for the other instruments.

### How \(\Gamma^*\) is Used in the Simulation:
Once you have \(\Gamma^*\), it is used to generate the observed data according to the equation:
\[
x_{i,t} = c_{i,t} \cdot \Gamma^* \cdot f_t + e_{i,t}
\]
Where:
- \(x_{i,t}\) is the observed data for entity \(i\) at time \(t\).
- \(c_{i,t}\) are the observable instruments for entity \(i\) at time \(t\).
- \(f_t\) are the latent factors at time \(t\).
- \(e_{i,t}\) is the error term (random noise).

In this equation, \(\Gamma^*\) determines how the latent factors \(f_t\) and the observable instruments \(c_{i,t}\) interact to generate the observed data \(x_{i,t}\).

### Summary of the Key Points:
- **\(\Gamma^*\)** is the true factor loadings matrix, defining the relationships between observable instruments and latent factors.
- **Size**: It is an \(L \times K\) matrix, where \(L\) is the number of instruments and \(K\) is the number of latent factors.
- **Random Generation**: It is generated using a normal distribution, meaning the relationships between instruments and factors are assumed to be normally distributed around 0, with a small spread (standard deviation = 0.1).
- **Purpose**: \(\Gamma^*\) is used to simulate the observed data, and the IPCA model's goal is to estimate a matrix \(\Gamma\) that is as close as possible to \(\Gamma^*\).

'''

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

''' Step 2: Run IPCA for Different Sample Sizes and Calculate Gamma Error

### 1. **Sample Sizes**
```python
sample_sizes = [10, 20, 50, 100, 200]
errors = []
```
- **`sample_sizes`**: This list defines the different values for \(N\) (the number of entities, e.g., stocks, firms, etc.) that you are testing. These sizes allow you to see how well the model estimates the latent factors and factor loadings as the sample size increases.
- **`errors`**: An empty list to store the error (the difference between the true \(\Gamma^*\) and estimated \(\Gamma\)) for each sample size.

### 2. **Loop Over Sample Sizes**
```python
for N in sample_sizes:
```
- You are looping over the different values of \(N\) (10, 20, 50, 100, and 200) to run the IPCA model for each sample size.

---

### 3. **Generate Simulated Data for Each Sample Size**
```python
X, f_t, Gamma_true, c_it = simulate_data(N, T, K, L, Gamma_true, sigma_e)
```
- **`simulate_data()`**: This function generates the simulated data for a given sample size \(N\).
  - **Inputs**: \(N\) (sample size), \(T\) (number of time periods), \(K\) (number of latent factors), \(L\) (number of observable instruments), \(\Gamma^*\) (true factor loadings), and \(\sigma_e\) (standard deviation of the error term).
  - **Outputs**:
    - `X`: The observed data for all entities over time.
    - `f_t`: The latent factors over time.
    - `Gamma_true`: The true factor loadings matrix (same for all sample sizes).
    - `c_it`: The observable instruments for each entity over time.

---

### 4. **Reshape the Data for the IPCA Model**

#### 4.1. Reshape the Observed Data
```python
X_flat = X.flatten()
```
- **`X_flat`**: The observed data \(X\) is reshaped into a 1D array.
  - **Before flattening**: \(X\) has dimensions \(N \times T\), where \(N\) is the number of entities and \(T\) is the number of time periods.
  - **After flattening**: \(X\) becomes a single vector of length \(N \times T\), where the observed values for each entity are concatenated across time.

#### 4.2. Reshape the Instruments
```python
c_it_flat = c_it.reshape(N * T, L)
```
- **`c_it_flat`**: The observable instruments matrix is reshaped into \(N \times T\) rows and \(L\) columns.
  - Each row corresponds to the instruments for one entity at a specific time point.
  - The total number of rows is \(N \times T\), where each row represents the instruments for a specific entity at a specific time period.
  - The number of columns is \(L\), representing the number of instruments (or observable characteristics).

---

### 5. **Create MultiIndex for Panel Data**
```python
entities = np.repeat(np.arange(N), T)  # Entity index
times = np.tile(np.arange(T), N)  # Time index

index = pd.MultiIndex.from_arrays([entities, times], names=['entity', 'time'])
```
- **Panel Data Structure**:
  - **`entities`**: This creates an array where each entity index (e.g., 0, 1, 2, ..., \(N-1\)) is repeated \(T\) times (once for each time period). This represents the entity IDs.
  - **`times`**: This creates an array where the time indices (0, 1, 2, ..., \(T-1\)) are repeated for each entity.
  - **`index`**: A **MultiIndex** is created from the `entities` and `times` arrays using `pandas`. This index will be used to label the data (both the instruments and the observed values) with the correct entity and time information.

---

### 6. **Convert Data to DataFrames with MultiIndex**
```python
c_it_df = pd.DataFrame(c_it_flat, index=index)
y_df = pd.Series(y_flat, index=index)
```
- **`c_it_df`**: The reshaped instruments `c_it_flat` are converted to a `pandas` DataFrame, where each row corresponds to a specific entity-time pair, indexed by the MultiIndex (`entity`, `time`).
- **`y_df`**: The reshaped observed data `y_flat` is converted to a `pandas` Series, similarly indexed by the entity-time MultiIndex.

---

### 7. **Fit the IPCA Model**
```python
regr = InstrumentedPCA(n_factors=K, intercept=False, alpha=alpha_reg, max_iter=max_iterations, iter_tol=tolerance, n_jobs=-1)
regr = regr.fit(X=c_it_df, y=y_df)
```
- **IPCA Model Setup**:
  - **`n_factors=K`**: The number of latent factors to estimate is set to \(K\).
  - **`intercept=False`**: No intercept is included in the model.
  - **`alpha=alpha_reg`**: Regularization parameter to prevent overfitting.
  - **`max_iter=max_iterations`**: Maximum number of iterations allowed for the alternating least squares (ALS) optimization procedure.
  - **`iter_tol=tolerance`**: Convergence tolerance; the model will stop iterating if the updates between iterations are smaller than this threshold.
  - **`n_jobs=-1`**: Parallelization; using all available CPU cores to speed up the computation.

- **`regr.fit()`**: This step fits the IPCA model using the observable instruments (`c_it_df`) and the observed data (`y_df`), estimating the latent factors and factor loadings.

---

### 8. **Retrieve the Estimated Factor Loadings \(\Gamma\)**
```python
Gamma_est, _ = regr.get_factors(label_ind=True)
```
- **`Gamma_est`**: The estimated factor loadings matrix \(\Gamma\) is retrieved after fitting the model. The model outputs:
  - **`Gamma_est`**: The estimated loadings matrix, which is an approximation of the true \(\Gamma^*\).
  - **`_`**: This would normally represent the estimated latent factors \(f_t\), but it's ignored here since you're only interested in the factor loadings.

---

### 9. **Calculate the Error**
```python
error = np.linalg.norm(Gamma_true - Gamma_est, ord='fro')**2
errors.append(error)
```
- **Error Calculation**:
  - **`Gamma_true - Gamma_est`**: The difference between the true loadings matrix \(\Gamma^*\) and the estimated loadings matrix \(\Gamma\).
  - **`np.linalg.norm(..., ord='fro')`**: This computes the **Frobenius norm** of the difference matrix, which is a measure of the total error between the true and estimated factor loadings.
    - The Frobenius norm is essentially the square root of the sum of the squared differences between all corresponding elements in the two matrices.
  - **`**2`**: Squaring the Frobenius norm gives you the squared error.
- **Store the Error**: The calculated error is appended to the `errors` list for later analysis.
'''

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

'''# Step 4: Plot ||Gamma - Gamma*||^2 vs log(N)
plt.plot(np.log(sample_sizes), errors, marker='o')
plt.title(r'$\| \Gamma - \Gamma^* \|^2$ vs Log(N)')
plt.xlabel('log(N)')
plt.ylabel(r'$\| \Gamma - \Gamma^* \|^2$')
plt.grid(True)
plt.show()'''


# Step 4: Plot ||Gamma - Gamma*||^2 vs log(N)
fig, ax1 = plt.subplots()

# Primary axis: log(N)
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