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

### Explanation and Interpretation of Results

---

### Data Dimensions

1. **X shape**: `(5290, 4)` - This shows that `X`, our characteristics matrix, has 5290 rows and 4 columns (characteristics: volatility, MA ratio, price-MA ratio, and volume). Each row corresponds to one observation (a specific time and stock).

2. **y shape**: `(5290,)` - This indicates `y`, our returns vector, also has 5290 entries matching the rows in `X`.

3. **indices shape**: `(5290, 2)` - The `indices` array, which tracks time and entity (stock ID) pairs, has the same number of rows (5290) with two columns (time and entity) for each entry.

---

### Fold 1

1. **Training Data Dimensions**:
   - **X_train shape**: `(4230, 4)` and **y_train shape**: `(4230,)` - In Fold 1, 4230 observations are used for training. We dropped some observations for cross-validation, so the dimensions are smaller than the full dataset.

2. **Fitting IPCA Model**:
   - **Convergence Steps**:
     - Convergence for Fold 1 takes 5 steps with progressively smaller aggregate updates, indicating the model is finding a stable solution.
   - **Estimated Gamma (Factor Loadings)**:
     - The Gamma matrix displays loadings for each characteristic on the three latent factors.
       - For example, characteristic 0 (volatility) loads heavily on Factor 1, meaning volatility is strongly related to Factor 1.
   - **Estimated Latent Factors**:
     - These factors summarize the relationships across stocks and are derived from the characteristics. For Fold 1, the latent factors are small but show slight variability, reflecting their calculated influence on returns.
   - **Variance Explained by Factors**:
     - **Proportion of Variance Explained**: Only **0.34%** of the total variance in `y_train` is explained by the factors in Fold 1, indicating that latent factors from these characteristics only partially capture returns variability.

---

### Fold 2

1. **Training Data Dimensions**:
   - Same dimensions as Fold 1, with 4230 observations used for training.

2. **Fitting IPCA Model**:
   - **Convergence Steps**: In Fold 2, convergence takes 6 steps. We again see a series of aggregate updates declining toward stability.
   - **Estimated Gamma (Factor Loadings)**:
     - The loadings in Fold 2 are different from Fold 1, as IPCA recalculates them for each fold. For example, characteristic 1 (MA ratio) has a large loading on Factor 2, suggesting it contributes more to Factor 2 in this fold.
   - **Variance Explained**: The variance explained by factors in Fold 2 is **0.45%**, slightly higher than in Fold 1 but still low, indicating limited predictive power.

---

### Fold 3

1. **Training Data Dimensions**: Same as Fold 1 and Fold 2.

2. **Fitting IPCA Model**:
   - **Convergence Steps**: Fold 3 converges in 4 steps.
   - **Gamma Matrix**:
     - Gamma loadings vary again in Fold 3, reflecting the differences in training data. Here, characteristic 0 (volatility) and characteristic 1 (MA ratio) load more on Factor 1.
   - **Variance Explained**: This time, the model explains **0.81%** of `y_train` variance, the highest observed so far, yet still modest.

---

### Fold 4

1. **Training Data Dimensions**: In Fold 4, there are 4235 observations (slightly more than in previous folds).

2. **Fitting IPCA Model**:
   - **Convergence Steps**: Fold 4 converges quickly in 3 steps.
   - **Gamma Matrix**:
     - Gamma loadings shift again in Fold 4, with characteristic 0 (volatility) showing a lower loading on Factor 1 and a stronger presence on Factor 2.
   - **Variance Explained**: The factors explain **0.97%** of the variance in `y_train`, slightly higher than previous folds, but still low.

---

### Fold 5

1. **Training Data Dimensions**: Same as Fold 4, with 4235 observations.

2. **Fitting IPCA Model**:
   - **Convergence Steps**: Fold 5 takes the longest to converge, with 14 steps, indicating a more complex fit.
   - **Gamma Matrix**:
     - In Fold 5, the Gamma matrix shows characteristic 2 (price-MA ratio) has the highest loading on Factor 3, suggesting its importance to this factor in this fold.
   - **Variance Explained**: The factors explain **0.48%** of the variance in `y_train`, indicating that predictive power remains limited across folds.

---

### Cross-Validation Summary: Procrustes Analysis

1. **Unaligned vs. Aligned Errors**:
   - **Average Unaligned Error**: `5.319066` - This measures the average difference between Gamma matrices across folds without alignment. High unaligned error suggests that Gamma matrices vary noticeably across folds.
   - **Average Aligned Error**: `1.543845` - After using Procrustes alignment (which accounts for rotation and scaling), the aligned error is significantly lower, showing that the Gamma matrices are more consistent in structure when aligned.
   - **Average Improvement**: **70.98%** - The alignment process provides a substantial reduction in error, meaning the model’s factor loadings are more consistent than they appear without adjustment.

---

### Interpretation of Results

1. **Model Stability**:
   - Despite variance in Gamma loadings across folds, Procrustes alignment reveals that the loadings retain a similar structure overall. This suggests that while the individual values vary, the relationships captured by IPCA are stable after adjusting for orientation differences.

2. **Predictive Power**:
   - The explained variance in each fold is very low (~0.3-0.9%), indicating that the latent factors derived from these characteristics explain only a small portion of the returns’ variability. This is common in financial data, where returns are influenced by many factors outside the chosen characteristics.

3. **Importance of Alignment**:
   - The large improvement in error post-alignment (over 70%) demonstrates that aligning factor loadings across folds is essential for ensuring the model’s results are interpretable and consistent. Without alignment, the observed variability might misleadingly suggest instability in the Gamma matrix.

In summary, the IPCA model’s factors offer some insights but have limited predictive power in explaining daily stock returns with the characteristics selected. Procrustes alignment helps to standardize results across folds, showing that while the values differ, the structure is relatively consistent.
