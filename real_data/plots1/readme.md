### Date: 19th November, 2025
### Code file is [here](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/simulations_7_8.py)

### **What Changes Were Made?**

1. **Reduced Factors**:
   - The number of latent factors tested (`n_factors`) is explicitly set to `[1, 2]` using the `factor_counts` list.
   - This ensures that the IPCA model is run separately for configurations with **1 factor** and **2 factors**, allowing us to compare their performance in explaining variance.

2. **Variance Explained Calculation**:
   - The variance explained by the latent factors is computed for each fold, and results are stored for both configurations.
   - This helps us understand how the number of factors impacts the explanatory power of the model.

3. **Updated Visualizations**:
   - Separate heatmaps are generated for the Gamma matrix (\( \Gamma \)) for both configurations, showcasing how characteristics load onto the factors.
   - A comparison plot is created to display the variance explained across folds for 1 factor versus 2 factors.

---

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

### **5. Aligned Error Heatmap (1 Factor)**

![aligned_error_heatmap_1_factors](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/plots1/aligned_error_heatmap_1_factors.png)

**Description**:
- This heatmap displays the aligned errors between factor loadings across folds after Procrustes alignment with **1 latent factor**.
- Each cell represents the Frobenius norm (a measure of difference) between the Gamma matrices of two folds after alignment.
- Diagonal values are zero because a fold is perfectly aligned with itself.

**Interpretation**:
- The low values across most cells (e.g., ~0.10–0.15) suggest that alignment significantly improves consistency between factor loadings across folds.
- Slightly higher errors are observed for Fold 1 vs. Fold 4 (0.15) and Fold 1 vs. Fold 3 (0.15), indicating these pairs are less consistent, even after alignment.
- Overall, alignment has successfully reduced discrepancies, ensuring that the latent factor is interpreted consistently across time.

---

### **6. Aligned Error Heatmap (2 Factors)**

![aligned_error_heatmap_2_factors](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/plots1/aligned_error_heatmap_2_factors.png)

**Description**:
- This heatmap shows the aligned errors for **2 latent factors** after Procrustes alignment.
- Values represent differences in factor loadings between folds, with larger values indicating greater discrepancies.

**Interpretation**:
- Errors are generally higher than in the 1-factor case, ranging from 0.65 to 1.5. This is expected because more factors introduce additional complexity and variation in the Gamma matrices.
- Fold 1 vs. Fold 4 (1.5) and Fold 1 vs. Fold 3 (1.4) exhibit the highest errors, suggesting that these folds have the most divergent factor loadings.
- Despite higher errors, alignment has still reduced discrepancies compared to unaligned errors, improving consistency across folds.

---

### **7. Percentage Difference Heatmap (Unaligned - Aligned) (1 Factor)**

![percentage_diff_heatmap_1_factors](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/plots1/percentage_diff_heatmap_1_factors.png)

**Description**:
- This heatmap visualizes the percentage reduction in error between unaligned and aligned Gamma matrices for **1 latent factor**.
- Positive values indicate improvement (alignment reduced error), while negative values suggest alignment increased error.

**Interpretation**:
- High improvements (e.g., ~97% for Fold 1 vs. Fold 2) demonstrate the effectiveness of alignment in reducing discrepancies.
- Negative values (e.g., -50% for Fold 2 vs. Fold 4) indicate rare cases where alignment slightly increased discrepancies, likely due to numerical instability or poor initial alignment.
- Overall, alignment leads to substantial error reduction across most fold comparisons, confirming its importance for interpretability.

---

### **8. Percentage Difference Heatmap (Unaligned - Aligned) (2 Factors)**

![percentage_diff_heatmap_2_factors](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/plots1/percentage_diff_heatmap_2_factors.png)

**Description**:
- This heatmap shows the percentage error reduction for **2 latent factors** after Procrustes alignment.

**Interpretation**:
- Most values are positive, indicating alignment significantly improves consistency, with reductions of up to 94% (Fold 1 vs. Fold 2).
- Some negative values (e.g., -16% for Fold 2 vs. Fold 4) suggest occasional misalignments or structural differences between folds that alignment cannot fully resolve.
- The overall improvements highlight that alignment is effective even with more complex configurations like 2 factors.

---

### **9. Unaligned Error Heatmap (1 Factor)**

![unaligned_error_heatmap_1_factors](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/plots1/unaligned_error_heatmap_1_factors.png)

**Description**:
- This heatmap displays the unaligned errors between Gamma matrices across folds for **1 latent factor** before alignment.
- Errors represent the raw differences in factor loadings without accounting for rotation or scaling.

**Interpretation**:
- Errors are relatively high (~3.9 for most fold pairs), showing significant discrepancies between Gamma matrices before alignment.
- These high unaligned errors confirm the need for Procrustes alignment to standardize the factor loadings across folds.

---

### **10. Unaligned Error Heatmap (2 Factors)**

![unaligned_error_heatmap_2_factors](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/plots1/unaligned_error_heatmap_2_factors.png)

**Description**:
- This heatmap shows the unaligned errors for **2 latent factors**, representing raw differences in Gamma matrices before alignment.

**Interpretation**:
- Errors are higher than in the 1-factor case (e.g., ~4–7), reflecting the added complexity of 2-factor configurations.
- Fold 4 vs. Fold 5 (7.7) has the highest error, suggesting these folds have the most divergent factor loadings.
- The high unaligned errors indicate that alignment is essential to ensure consistent interpretation of factor loadings across time.

---

### Overall Insights

1. **Effectiveness of Procrustes Alignment**:
   - The aligned error heatmaps and percentage difference heatmaps confirm that Procrustes alignment significantly reduces discrepancies across folds, improving the stability of factor loadings.
   - Alignment is particularly effective in the 1-factor case, where most errors are reduced to near-zero values.

2. **Complexity with 2 Factors**:
   - Errors are consistently higher in the 2-factor case, both before and after alignment, highlighting the increased complexity of interpreting multiple latent factors.

3. **Limitations of Alignment**:
   - Occasional negative percentage differences suggest that alignment may not fully resolve structural differences in certain fold pairs, especially for higher factor counts or divergent Gamma matrices.

These plots reinforce the importance of alignment for ensuring consistency in factor loadings and highlight the trade-off between interpretability and complexity when increasing the number of latent factors. Let me know if further analysis is required!
