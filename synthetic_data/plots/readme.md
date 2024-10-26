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

### Summary and Conclusions

1. **Effectiveness of Procrustes Alignment**: Procrustes alignment consistently reduces error by around 90% across configurations, minimizing rotational ambiguity and stabilizing error values.

2. **Key Patterns**:
   - **Aligned Error Stability**: Aligned errors are consistently low across all configurations, indicating Procrustes alignment’s robustness.
   - **Unaligned Error Sensitivity**: Unaligned errors are higher and more sensitive to noise variance and sample size.

3. **Value of Alignment**: The contrast between aligned and unaligned errors emphasizes Procrustes alignment’s role in reliable factor loading estimation by mitigating rotational ambiguity.

### Outstanding Issues and Further Directions

- **Sensitivity to Extreme Noise**: Though Procrustes alignment reduces error, aligned errors show slight dependence on noise and sample size. Additional regularization or preprocessing may help mitigate this.
- **Larger Sample Sizes**: Exploring larger sample sizes could confirm if aligned error remains stable under different conditions.
- **Enhanced Alignment Techniques**: Further improvements could involve advanced alignment methods or domain-specific transformations to boost accuracy and robustness.

**Conclusion**: Procrustes alignment significantly improves factor loading accuracy by reducing error across configurations. However, slight sensitivity to high noise suggests areas for enhancement.

### Lack of Clear Pattern in Error Trends

From the heatmaps and plots, there is no consistent trend showing that error between Gamma_{{true} and estimated Gamma (both aligned and unaligned) decreases as sample size N increases and noise variance sigma^2 decreases. This lack of pattern may be due to the following:

### Potential Causes

1. **Noise Interference in Factor Estimation**:
   - **High Noise Variability**: High noise variance (sigma^2) can obscure trends by introducing randomness that overrides sample size effects.
   - **Impact on Latent Factor Accuracy**: High noise levels affect observed data, impacting IPCA’s ability to accurately estimate latent factors, especially when N increases alone.

2. **Sample Size Insufficiency**:
   - **Limited Sample Sizes**: The sample sizes may still be too low to fully capture latent structures, especially under high noise conditions, leading to variable error outcomes.
   - **Nonlinear Data Relationships**: For complex data relationships, increasing N alone may not improve alignment if the signal-to-noise ratio remains low.

3. **Rotational Ambiguity Effects**:
   - **Persistent Ambiguity**: Procrustes alignment corrects rotational ambiguity post-estimation, yet IPCA’s factor estimates may still vary due to this ambiguity, especially with high noise.
   - **Dependence on Factor Structure**: Estimation accuracy depends on the underlying latent structure, which may not be fully captured in the presence of noise and limited sample sizes.

4. **IPCA Model Limitations**:
   - **Sensitivity to Hyperparameters**: IPCA performance may vary depending on factor count K and instrument number L. If these parameters don’t align with the true data process, increasing N may not consistently reduce error.
   - **Lack of Regularization**: Without regularization alpha_{reg}} = 0, the model may overfit, leading to inconsistent patterns across different N and sigma^2 values.

5. **Finite Sample Variability**:
   - **High Variability in Small Samples**: Smaller sample sizes contribute to higher (Gamma) variability, resulting in inconsistent patterns. Some sample configurations may perform better by chance, despite noise presence.

### Final Conclusion

- **Inconsistent Error Patterns**: We do not observe a clear reduction in |Gamma_{true} - Gamma| with increasing N and decreasing (sigma^2).
- **Main Contributing Factors**: High noise, finite sample effects, rotational ambiguity, and IPCA limitations collectively contribute to the observed variability in error.
- **Suggested Improvements**: To clarify trends, we could test larger sample sizes, adjust model parameters (e.g., regularization), or consider alternative alignment methods. A more robust experimental design focusing on noise and sample size sensitivity could provide further insights.
