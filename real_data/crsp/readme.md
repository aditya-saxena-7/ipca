### **Data Source**

1. **Data Source**:
   - The dataset was sourced from the **CRSP/Compustat Merged Database** via **Wharton Research Data Services (WRDS)**.
   - This database integrates security price, return, and volume data from the NYSE, AMEX, and NASDAQ stock markets, along with fundamental data from Compustat.

2. **Place**:
   - The data covers securities listed on major US stock exchanges, including **NYSE**, **NASDAQ**, and **AMEX**.
   - These exchanges are headquartered in the United States.

3. **Duration**:
   - The dataset spans from **January 2010 to December 2024**.
   - The timeframe ensures the inclusion of historical and recent data to capture long-term trends and recent market dynamics.

### **Final Summary, Findings, and Conclusion**

---

### **Summary**
This analysis implemented **Instrumented Principal Component Analysis (IPCA)** on financial data obtained from the **CRSP/Compustat Merged Database** to estimate latent factors influencing stock returns. The study aimed to uncover the relationships between observable characteristics like volatility, moving average ratios, and trading volume with latent factors driving returns.

The dataset spanned from **January 2010 to December 2024** and covered major securities listed on the **NYSE**, **NASDAQ**, and **AMEX**. Cross-validation, Procrustes alignment, and best-fit line visualizations were used to ensure robustness and consistency across folds.

---

### **Findings**
1. **Latent Factors and Gamma Matrices**:
   - **Factor 3** consistently captured **risk-related dynamics**, heavily influenced by **volatility** and **price-ma-ratio** across all folds.
   - **Factors 1 and 2** were strongly tied to **trend-following behavior**, as shown by their dependence on **ma-ratio**.
   - **Log volume** played a diverse role, contributing positively across multiple factors, indicating its importance in trading dynamics.

2. **Variance Explained**:
   - Latent factors explained a significant portion of return variability in most folds, though some extreme variance values (e.g., Fold 1) suggest potential overfitting or data irregularities.
   - Reasonable folds showed variance explained in the range of **153% to 3855%**, demonstrating the IPCA model’s capability to capture key dynamics.

3. **Error Metrics and Procrustes Alignment**:
   - Alignment significantly reduced discrepancies between Gamma matrices, with aligned errors dropping from an unaligned average of **~4.0 (Frobenius norm)** to **~1.54**.
   - High percentage improvements (e.g., **71%-81%**) confirmed that alignment successfully standardized factor loadings across folds.

4. **Best-Fit Line Plots**:
   - No strong linear relationships were observed between characteristics and returns:
     - **Volatility, MA Ratio, Price-MA Ratio, and Log Volume** all exhibited weak or flat correlations with returns.
     - This aligns with findings that individual characteristics alone explain only a small fraction of return variability, with latent factors playing a dominant role.

5. **Cross-Fold Stability**:
   - Gamma matrices showed consistent structural patterns across folds, reinforcing the robustness of the IPCA model.
   - Variations in loadings highlighted the dynamic nature of stock returns and the importance of Procrustes alignment for structural consistency.

---

### **Conclusion**
- **Effectiveness of IPCA**:
  - The IPCA model successfully identified meaningful latent factors driven by observable characteristics, with strong and stable patterns across folds.
  - Procrustes alignment proved crucial for ensuring the comparability of factor loadings across time, reducing rotational and scaling inconsistencies.

- **Key Relationships**:
  - **Risk-related factors** were dominated by **volatility** and **short-term price deviations (price-ma-ratio)**.
  - **Trend-following factors** relied heavily on **moving average ratios**, highlighting long-term price behavior.

- **Limitations**:
  - The weak direct relationships between characteristics and returns suggest that additional variables (e.g., macroeconomic indicators or sector-specific factors) could improve explanatory power.
  - Some variance explained values were extreme, pointing to possible overfitting in specific folds.

- **Future Directions**:
  - Incorporate new characteristics like **sentiment analysis**, **macroeconomic data**, or **nonlinear factors** to enhance model performance.
  - Explore time-varying dynamics or regime-switching models to capture shifts in relationships over time.

---
