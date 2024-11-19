### Date: 19th November, 2025
### Code file is [here]()

### 1. **Gamma Matrix Heatmap - Fold 5 (1 Factor)** (`gamma_heatmap_fold_5_1_factors.png`)

![gamma_heatmap_fold_5_1_factors](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/plots1/gamma_heatmap_fold_5_1_factors.png)

**Description**:
- This heatmap shows the Gamma matrix for Fold 5 with 1 latent factor. The rows represent the characteristics (volatility, MA ratio, price-MA ratio, and volume), and the single column (Factor 0) represents the latent factor.
- The values in the heatmap indicate the strength and direction (positive or negative) of each characteristic's contribution to the latent factor.

**Interpretation**:
- **Characteristic 2 (Price-MA Ratio)** has the strongest positive loading (0.72) on the single latent factor. This suggests that short-term price movements relative to moving averages heavily influence this factor.
- **Characteristic 0 (Volatility)** has a moderately negative loading (-0.47), indicating that volatility negatively contributes to the factor.
- **Characteristics 1 (MA Ratio)** and **3 (Volume)** also contribute to the factor but with lower magnitudes.
- Overall, this heatmap suggests that the latent factor in Fold 5 is driven primarily by the Price-MA Ratio, with some influence from volatility.

---

### 2. **Gamma Matrix Heatmap - Fold 5 (2 Factors)** (`gamma_heatmap_fold_5_2_factors.png`)

![gamma_heatmap_fold_5_2_factors](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/plots1/gamma_heatmap_fold_5_2_factors.png)

**Description**:
- This heatmap displays the Gamma matrix for Fold 5 with 2 latent factors. Rows represent characteristics, and columns represent the two latent factors (Factor 0 and Factor 1).
- The values show the strength and direction of each characteristic's contribution to each factor.

**Interpretation**:
- **Factor 0**:
  - **Characteristic 2 (Price-MA Ratio)** has the strongest positive loading (0.75), making it the primary driver of this factor.
  - **Characteristic 3 (Volume)** also contributes positively (0.42), indicating its secondary importance in Factor 0.
- **Factor 1**:
  - **Characteristic 0 (Volatility)** has a high positive loading (0.73), suggesting that this factor captures risk-related dynamics in stock returns.
  - **Characteristic 1 (MA Ratio)** contributes moderately (0.45), indicating its influence on long-term trends.
- The heatmap highlights the division of labor between the two factors: Factor 0 focuses on short-term price deviations (Price-MA Ratio), while Factor 1 emphasizes risk and trend metrics (Volatility and MA Ratio).

---

### 3. **Variance Explained by Factors Across Folds** (`variance_explained_comparison.png`)

![variance_explained_comparison](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/plots1/variance_explained_comparison.png)

**Description**:
- This line plot compares the percentage of variance in stock returns explained by 1 vs. 2 latent factors across all 5 folds.
- The x-axis represents the fold number, and the y-axis shows the proportion of variance explained (%).

**Interpretation**:
- The orange line (2 factors) consistently explains more variance than the blue line (1 factor) across all folds, as expected since adding a second factor increases the model's flexibility.
- The variance explained by 2 factors peaks at Fold 4 (~0.8%), while 1 factor peaks at Fold 3 (~0.4%). However, the overall proportion of variance explained remains low in both cases, suggesting that the selected characteristics capture only a small fraction of stock return variability.
- This result emphasizes the complexity of financial return data, where other unobserved factors likely play a significant role.

---

### 4. **Variance Explained Summary** (`variance_explained_summary.csv`)

**Description**:
- This CSV file contains a summary of the variance explained by 1 and 2 factors across all folds.
- The columns include:
  - **Factors**: Number of latent factors (1 or 2).
  - **Avg_Variance_Explained**: The average percentage of variance explained across all folds.
  - **Variance_Explained_Per_Fold**: A list of variance explained values for each fold.

**Interpretation**:
- The CSV allows for a direct comparison between the average performance of 1 vs. 2 factors:
  - **1 Factor**: Explains ~0.3-0.4% variance on average.
  - **2 Factors**: Explains ~0.6-0.8% variance on average, showing a clear improvement over 1 factor.
- While 2 factors perform better, the overall explained variance remains low, suggesting that additional characteristics or alternative modeling techniques might be needed to capture more of the variability in returns.

---

### Key Insights from the Results
1. **Role of Characteristics**:
   - The Price-MA Ratio (short-term price deviations) consistently shows strong positive loadings on the primary factor(s), indicating its importance in explaining stock return variability.
   - Volatility also plays a significant role, particularly in Factor 1 when 2 factors are used, suggesting that risk is a critical driver of returns.

2. **Impact of Adding Factors**:
   - Adding a second factor increases the explained variance, especially for folds where single-factor models underperform. However, the improvement is modest, highlighting the limitations of the selected characteristics in fully capturing stock return variability.

3. **Limitations of the Model**:
   - The low overall variance explained (even with 2 factors) suggests that other unobserved factors or dynamics may drive returns. Expanding the set of characteristics or exploring nonlinear relationships could improve the model’s performance.
