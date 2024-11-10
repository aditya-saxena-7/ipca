### 1. Aligned Error Heatmap (`aligned_error_heatmap.png`)

![aligned_error_heatmap](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/plots/aligned_error_heatmap.png)

**Description**: This heatmap shows the aligned error between the `Gamma` matrices across different folds after applying Procrustes alignment. Each cell in the matrix represents the error between two folds, with lower values indicating higher similarity after alignment.

**Interpretation**:
- The diagonal values are zero because each fold is compared with itself, resulting in no alignment error.
- Off-diagonal values generally remain low (around 1.2 to 2.1), suggesting that Procrustes alignment has effectively standardized the factor loadings across folds.
- Lower aligned errors between folds (like 1.2 between Fold 2 and Fold 5) indicate more consistency in factor loadings between these folds, while higher values (like 2.1 between Fold 1 and Fold 5) suggest that Fold 5’s loadings differ more significantly from Fold 1.
- Overall, the aligned errors are relatively low, indicating that the structure of the Gamma matrices is stable across different folds once alignment is applied.

---

### 2. Alignment Comparison (`alignment_comparison.png`)

![alignment_comparison](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/plots/alignment_comparison.png)

**Description**: This bar chart compares unaligned and aligned errors for each pair of folds. The blue bars represent unaligned errors, and the orange bars represent aligned errors.

**Interpretation**:
- The unaligned errors (blue) are consistently higher than the aligned errors (orange), emphasizing the improvement achieved through Procrustes alignment.
- The significant reduction in error (blue to orange) across most pairs indicates that alignment corrects for variations in orientation and scaling between folds, making the factor loadings more consistent.
- The stark contrast between unaligned and aligned errors visually confirms the importance of using alignment to interpret factor loadings reliably across cross-validation folds.

---

### 3. Correlation Heatmap of Characteristics (`characteristics_correlation_heatmap.png`)

![characteristics_correlation_heatmap](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/plots/characteristics_correlation_heatmap.png)

**Description**: This heatmap shows the correlations between the different characteristics (volatility, MA ratio, price-MA ratio, and volume).

**Interpretation**:
- **Volatility** and **volume** have a moderate positive correlation (0.48), suggesting that stocks with higher trading volumes also tend to have more volatile returns.
- **Volatility** and **price-MA ratio** have a negative correlation (-0.55), indicating that higher volatility often corresponds with prices being further away from their recent moving averages.
- **MA ratio** is weakly correlated with the other characteristics, with low values like -0.27 with volatility and -0.035 with volume. This indicates that the MA ratio might capture independent trend information, separate from volatility or volume dynamics.
- This correlation matrix helps us understand how characteristics interact and which might influence latent factors similarly in the IPCA model.

---

### 4. Pairplot of Characteristics (`characteristics_pairplot.png`)

![characteristics_pairplot](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/plots/characteristics_pairplot.png)

**Description**: This pairplot provides scatterplots for each characteristic against every other characteristic, along with histograms on the diagonal representing the distribution of each characteristic.

**Interpretation**:
- The histograms on the diagonal show the distribution of each characteristic. For example, **volatility** has a right-skewed distribution, while **volume** is approximately normal.
- The scatter plots between characteristics reveal the relationships seen in the correlation heatmap:
  - The negative relationship between **volatility** and **price-MA ratio** is visible, with higher volatility values corresponding to lower price-MA ratios.
  - **MA ratio** shows no strong visual trend with other characteristics, supporting its low correlations seen earlier.
- This plot provides a detailed view of how characteristics vary relative to each other, helping us assess their individual contributions to the IPCA model.

---

### 5. Error Difference Heatmap (Unaligned - Aligned) (`error_difference_heatmap.png`)

![error_difference_heatmap](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/plots/error_difference_heatmap.png)

**Description**: This heatmap displays the difference between unaligned and aligned errors for each fold comparison. Higher values indicate greater improvement due to alignment.

**Interpretation**:
- The diagonal is zero, as comparing each fold with itself results in no difference.
- High values, such as 8.2 (Fold 1 vs. Fold 4), indicate that alignment had a substantial impact, greatly reducing discrepancies between the factor loadings in these folds.
- Negative or near-zero values suggest that for these fold comparisons, alignment had minimal or even negligible effect. For instance, Fold 4 vs. Fold 2 shows -0.03, meaning alignment did not significantly improve similarity in this pair.
- This heatmap highlights where alignment is most impactful, confirming that Procrustes alignment is essential for accurate interpretation of factor loadings across different cross-validation folds.

---

### 6. Gamma Matrix Heatmap - Fold 5 (`gamma_heatmap_fold_5.png`)

![gamma_heatmap_fold_5](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/plots/gamma_heatmap_fold_5.png)

**Description**: This heatmap displays the estimated `Gamma` matrix for Fold 5, showing the loadings of each characteristic on the three latent factors.

**Interpretation**:
- **Characteristic 2 (price-MA ratio)** has a strong positive loading on Factor 0 (0.72) and a negative loading on Factor 1 (-0.53), suggesting it contributes significantly to Factor 0.
- **Characteristic 3 (volume)** has a high positive loading on Factor 2 (0.87), indicating that volume heavily influences this factor in Fold 5.
- This Gamma matrix shows which characteristics have the most substantial influence on each latent factor in Fold 5. Differences across folds would highlight which relationships are stable versus variable.

---

### 7. Unaligned Error Heatmap (`unaligned_error_heatmap.png`)

![unaligned_error_heatmap](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/plots/unaligned_error_heatmap.png)

**Description**: This heatmap shows the unaligned errors between `Gamma` matrices across folds. It represents the difference between the `Gamma` matrices without applying Procrustes alignment.

**Interpretation**:
- Larger values (e.g., 9.5 between Fold 1 and Fold 4) indicate more significant discrepancies in factor loadings between those folds.
- Compared to the aligned error heatmap, these unaligned errors are generally higher, suggesting that factor loadings appear inconsistent when alignment isn’t applied.
- This plot reinforces the necessity of alignment for cross-validation in IPCA, as unaligned errors tend to exaggerate differences between folds that are, in reality, structurally similar once aligned.

---

### 8. Variance Explained by Factors in Each Fold (`variance_explained.png`)

![variance_explained](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/plots/variance_explained.png)

**Description**: This bar chart shows the percentage of variance in `y_train` (returns) explained by the estimated factors in each fold.

**Interpretation**:
- Variance explained ranges from about 0.3% in Fold 1 to nearly 1% in Fold 4, indicating that the latent factors derived from our characteristics (volatility, MA ratio, etc.) have limited predictive power for daily stock returns.
- While the low values are typical for financial data (where returns are influenced by various external and stochastic factors), it highlights that the chosen characteristics may not fully capture the factors driving returns in this dataset.
- This plot suggests that while IPCA provides some explanatory power, adding more characteristics or considering different ones might improve the model’s performance.
