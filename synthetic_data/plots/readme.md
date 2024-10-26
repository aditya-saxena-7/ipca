### 1. **Avg_Aligned_Error_heatmap.png**

![Avg_Aligned_Error_heatmap](https://github.com/aditya-saxena-7/ipca/blob/main/synthetic_data/plots/20241026_133634/Avg_Aligned_Error_heatmap.png)

**Description**: This heatmap shows the average aligned error (| Gamma_{true} - Gamma_{aligned} |^2) for various noise variances (sigma^2) and sample sizes (N).

**Interpretation**:
- **Low and Consistent Error**: After applying Procrustes alignment, the error values remain consistently low across different sample sizes and noise levels, mostly around 0.29–0.31.
- **Effect of Noise Variance**: The errors show minor variations with noise, indicating that Procrustes alignment reduces sensitivity to noise variance. However, at higher noise levels (e.g., sigma^2 = 0.05), there is a slight drop in aligned error, particularly for smaller sample sizes (N = 10, 40).

**Significance**:
- This plot demonstrates the effectiveness of Procrustes alignment in stabilizing the estimation error across varying conditions, making the model robust to different sample sizes and noise levels.
- **Conclusion**: The alignment process successfully minimizes rotational ambiguity, leading to stable error values.

---

### 2. **Avg_Unaligned_Error_heatmap.png**

![Avg_Unaligned_Error_heatma](https://github.com/aditya-saxena-7/ipca/blob/main/synthetic_data/plots/20241026_133634/Avg_Unaligned_Error_heatmap.png)

**Description**: This heatmap shows the average unaligned error (| Gamma_{true} - Gamma_{est} |^2) without applying Procrustes alignment.

**Interpretation**:
- **Higher Errors and Variability**: The unaligned errors are significantly higher than the aligned errors, with values ranging from approximately 2.7 to 3.6.
- **Sensitivity to Noise and Sample Size**: The unaligned error is sensitive to both noise variance and sample size:
  - **Higher Noise**: Higher noise variances, such as \(\sigma^2 = 0.05\), lead to larger unaligned errors.
  - **Sample Size Effect**: Increasing the sample size slightly reduces unaligned error in some cases, but it remains inconsistent.

**Significance**:
- **Rotational Ambiguity**: The high variability in unaligned error highlights the rotational ambiguity issue in factor estimation models. Procrustes alignment is crucial here to achieve accurate factor loading estimates.
- **Conclusion**: Without alignment, the error values vary significantly, indicating that Procrustes alignment is essential for achieving stable and accurate loadings.

---

### 3. **error_difference_heatmap.png**

![error_difference_heatmap](https://github.com/aditya-saxena-7/ipca/blob/main/synthetic_data/plots/20241026_133634/error_difference_heatmap.png)

**Description**: This heatmap visualizes the difference between unaligned and aligned errors for each combination of \(\sigma^2\) and N.

**Interpretation**:
- **Significant Error Reduction**: The difference between unaligned and aligned errors is substantial, ranging from approximately 2.4 to 3.3 across configurations.
- **Higher Impact with Higher Noise and Smaller Samples**: The largest differences (error reduction) appear at higher noise levels (sigma^2 = 0.05) and smaller sample sizes (e.g., N = 10, 40). This suggests that Procrustes alignment is particularly effective in challenging conditions.

**Significance**:
- **Improvement Due to Alignment**: This plot highlights the benefit of Procrustes alignment in reducing estimation error by correcting for rotational misalignment. 
- **Conclusion**: Procrustes alignment drastically reduces the error, especially when the model faces higher noise or smaller sample sizes, showcasing its effectiveness.

---

### 4. **log_alignment_comparison.png**

![log_alignment_comparison](https://github.com/aditya-saxena-7/ipca/blob/main/synthetic_data/plots/20241026_133634/log_alignment_comparison.png)

**Description**: This plot compares the log of the unaligned and aligned errors against the log of sample size (N).

**Interpretation**:
- **Aligned Error Stability**: The aligned error remains consistently low (around -1 in log scale) regardless of sample size, showcasing the effectiveness of alignment in maintaining low errors.
- **Unaligned Error Variability**: The unaligned error stays above 1 on the log scale, with minor fluctuations across sample sizes. This demonstrates that, without alignment, the model’s error remains significantly higher and more variable.

**Significance**:
- **Procrustes Analysis Impact**: The stable, low aligned error line emphasizes the alignment’s role in reducing and stabilizing error. 
- **Conclusion**: Alignment effectively controls for variability and error magnitude, especially when dealing with changes in sample size.

---

### 5. **percentage_improvement_heatmap.png**

![percentage_improvement_heatmap](https://github.com/aditya-saxena-7/ipca/blob/main/synthetic_data/plots/20241026_133634/percentage_improvement_heatmap.png)

**Description**: This heatmap shows the percentage improvement in error due to Procrustes alignment for each combination of sigma^2 and N.

**Interpretation**:
- **Consistent High Improvement**: The percentage improvement remains high across all configurations, generally around 89%–91%. This indicates that alignment consistently provides significant error reduction.
- **Slightly Higher Improvement in Challenging Conditions**: The improvement tends to be slightly higher at higher noise levels (e.g., sigma^2 = 0.05) and smaller sample sizes. This is consistent with the `error_difference_heatmap`, where alignment had the most impact under these conditions.

**Significance**:
- **Robustness of Alignment**: This plot underscores the alignment’s importance and robustness in enhancing model accuracy, as it consistently provides around 90% improvement.
- **Conclusion**: Procrustes alignment is an effective solution for minimizing error in factor loading estimates, especially when data conditions are less ideal (e.g., high noise, small sample sizes).

---

### Overall Summary and Conclusions

1. **Effectiveness of Procrustes Alignment**: The Procrustes alignment consistently reduces error by around 90% across various configurations, stabilizing error values and minimizing rotational ambiguity.
2. **Trends and Patterns**:
   - **Aligned Error Stability**: Aligned errors remain low and stable across all configurations, demonstrating the robustness of Procrustes alignment.
   - **Unaligned Error Sensitivity**: Unaligned errors are significantly higher and more sensitive to changes in noise variance and sample size.
3. **Importance of Alignment**: The comparison between aligned and unaligned errors highlights Procrustes alignment’s role in ensuring reliable factor loading estimation by mitigating rotational ambiguity.

### Remaining Issues and Further Improvements

- **Sensitivity to Extreme Noise**: While Procrustes alignment effectively reduces error, there is a slight dependency on noise variance and sample size even in the aligned errors. Further exploration of additional regularization or data preprocessing methods might reduce this minor dependency.
- **Higher Sample Sizes**: The impact of larger sample sizes could be further explored to confirm if the aligned error consistently remains stable.
- **Enhanced Alignment Techniques**: Investigating advanced alignment techniques or incorporating domain-specific transformations might further improve accuracy and robustness.

**Conclusion**: Procrustes alignment significantly enhances the accuracy of factor loading estimation, as demonstrated by the consistent and substantial error reduction across various configurations. However, slight sensitivity to extreme conditions indicates potential areas for further enhancement.
