### Please find the working code [here](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/simulations_7_6.py)

### Dataset and Characteristics

1. **Data Source**: This code uses daily stock data from Yahoo Finance for five major tech companies (Apple, Microsoft, Google, Amazon, and Meta) from January 1, 2015, to December 31, 2019. We download adjusted close prices and trading volumes for each of these stocks.

2. **Daily Returns Calculation**: We compute daily returns for each stock, which represent the percentage change in price from one day to the next. This gives us a time series of returns, which is essential in understanding stock behavior over time.

3. **Characteristics Creation**:
   - **Volatility**: Calculated as a rolling 20-day standard deviation of returns, indicating the degree of price variation over this period. Higher values mean more variability in returns.
   - **MA Ratio (Moving Average Ratio)**: This is the ratio of the 50-day moving average to the 200-day moving average, capturing long-term trends. A higher value typically indicates upward momentum.
   - **Price-to-MA Ratio**: This ratio measures how the current price compares to the 50-day moving average. It helps assess if the price is above or below recent trends.
   - **Volume**: Log-transformed average daily volume, representing the number of shares traded. Higher values suggest more active trading.

4. **Summary Statistics**:
   - **Returns**: Mean returns across these stocks are positive but very small, indicating that daily gains are generally slight. Standard deviations (e.g., Apple: ~1.5%) show daily return variability.
   - **Characteristics**: For example, `volatility` shows a mean of ~0.0145, suggesting daily returns vary within a low range. `ma_ratio` and `price_ma_ratio` are both near 1 on average, meaning prices hover around moving averages without drastic deviations.

### Running IPCA on Each Fold (Cross-Validation)

**Purpose**: Cross-validation with IPCA (Instrumented Principal Component Analysis) assesses how well our estimated factors (latent characteristics) explain returns. Splitting the data into folds and running IPCA on each allows us to test the model’s reliability and consistency.

1. **Cross-Validation Setup**: We divide the time dimension of the data into 5 folds (or sections). For each fold, we use one part of the data as the validation set and the remaining as the training set.
   
2. **Data Structure**: 
   - We organize characteristics (`X`) and returns (`y`) in a panel format, where each row represents a specific stock at a specific time. The `indices` array keeps track of these time-stock pairs for easy reference.

3. **Training Data Selection**: For each fold, we create masks to select the training samples, where `X_train` is the training characteristics and `y_train` is the target variable (returns). We repeat this process to ensure all data is used across folds.

### Fit IPCA and Convergence Parameters

1. **IPCA Setup and Fitting**: 
   - We define IPCA with three latent factors (our hidden variables that aim to explain returns), no regularization (`alpha=0`), and a maximum of 1000 iterations for fitting.
   - The IPCA algorithm iteratively estimates factors and loadings for each stock characteristic and converges when updates between iterations fall below a threshold (`iter_tol=1e-3`).

2. **Convergence**: Convergence indicates that the model has stabilized in estimating factors and loadings for each fold. The number of iterations it takes to converge varies across folds and is printed out during fitting.

### Estimated Parameters and Gamma Matrix (Factor Loadings)

1. **Estimated Gamma Matrix**: `Gamma` represents the loadings of each characteristic on the latent factors. Each row corresponds to a characteristic (e.g., `volatility`, `ma_ratio`), and each column represents a latent factor. Higher values in the Gamma matrix indicate a stronger relationship between a characteristic and a latent factor.
   
   - For instance, in Fold 1, volatility has high loadings on Factor 1, meaning volatility might heavily contribute to that factor.

2. **Latent Factors**: The `factors` are estimated hidden variables derived from the characteristics, which IPCA uses to explain returns. These are essential to the IPCA model since they summarize the underlying relationships across stocks.

3. **Variance Explained**: For each fold, we calculate the proportion of variance in `y_train` explained by the estimated factors. This shows how much of the variation in returns is captured by the latent factors. In this dataset, it’s quite low (~0.34% in Fold 1, increasing up to ~0.97%), meaning that while some variation is captured, much of it is left unexplained, likely due to the complexity of financial data.

### Procrustes Analysis

1. **Purpose of Procrustes Analysis**: After obtaining the Gamma matrices for each fold, we need to check the consistency of factor loadings across folds. Since factor loadings (Gamma matrices) might be similar in structure but vary in scale or orientation, we align these matrices using Procrustes analysis.

2. **Unaligned vs. Aligned Errors**: 
   - **Unaligned Error**: Measures the difference between Gamma matrices across folds without any adjustment. Higher values indicate more significant discrepancies.
   - **Aligned Error**: Measures the difference after Procrustes alignment, which adjusts for rotation and scaling differences. Reduced aligned errors indicate better consistency in loadings across folds.
   
3. **Improvement from Alignment**: The difference between unaligned and aligned errors shows the effectiveness of Procrustes alignment. In our cross-validation summary, the alignment reduced error by an average of ~71%, meaning the Gamma matrices are more consistent across folds after alignment.

### Summary of Results

1. **Average Errors**: 
   - The unaligned and aligned errors give insight into how much IPCA’s Gamma matrices vary across folds. Lower aligned errors mean the model’s factor loadings are more consistent after adjustment.

2. **Variance Explained**: The IPCA model explains only a small proportion of return variability (~0.3-0.9%) in this dataset, suggesting the latent factors derived from these characteristics aren’t strong predictors of daily returns. This low variance explained is typical in financial data, where returns are influenced by a broad set of unpredictable factors.
