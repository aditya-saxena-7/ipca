### EDA Results Interpretation for Inclusion in the Paper

#### **8. Exploratory Data Analysis**

This section provides insights into the input data's structure and relationships through a detailed exploratory data analysis (EDA). Statistical summaries, correlation heatmaps, and visualizations of characteristics and returns are examined to understand the dataset's dynamics.

---

### **8.1 Statistical Summaries**

#### **8.1.1 Adjusted Close Prices**
| Statistic        | AAPL  | AMZN  | GOOGL | META  | MSFT  |
|------------------|-------|-------|-------|-------|-------|
| **Mean**         | 36.19 | 55.72 | 46.22 | 142.59 | 74.84 |
| **Std**          | 11.12 | 26.59 | 11.26 | 37.95 | 31.70 |
| **Min**          | 20.69 | 14.35 | 24.79 | 73.83 | 34.69 |
| **Max**          | 70.66 | 101.98| 67.96 | 216.85| 152.04|

- **Interpretation**:
  - The adjusted close prices show a wide range, with Meta (META) having the highest average price and Apple (AAPL) the lowest.
  - Amazon (AMZN) exhibits the highest variability in price (standard deviation of ~26.59), reflecting its volatile market behavior during this period.
  - The minimum values suggest significant dips in stock prices, potentially during broader market corrections or company-specific downturns.

---

#### **8.1.2 Daily Returns**
| Statistic        | AAPL    | AMZN    | GOOGL   | META    | MSFT    |
|------------------|---------|---------|---------|---------|---------|
| **Mean**         | 0.0011  | 0.0013  | 0.0007  | 0.0009  | 0.0013  |
| **Std**          | 0.0154  | 0.0176  | 0.0141  | 0.0181  | 0.0141  |
| **Min**          | -0.0996 | -0.0782 | -0.0750 | -0.1896 | -0.0717 |
| **Max**          | 0.0704  | 0.1322  | 0.0962  | 0.1552  | 0.1008  |

- **Interpretation**:
  - The mean returns are positive but close to zero, as expected for daily stock returns over long periods.
  - Meta shows the highest maximum daily return (0.1552) and the lowest minimum (-0.1896), suggesting high intraday volatility.
  - Microsoft (MSFT) and Google (GOOGL) exhibit more stable returns, as reflected in their lower standard deviations.

---

#### **8.1.3 Characteristics**
| Statistic        | Volatility | MA Ratio | Price-MA Ratio | Volume  |
|------------------|------------|----------|----------------|---------|
| **Mean**         | 0.0145     | 1.0688   | 1.0239         | 17.851  |
| **Std**          | 0.0058     | 0.0471   | 0.0432         | 0.3463  |
| **Min**          | 0.0046     | 0.9149   | 0.8516         | 16.756  |
| **Max**          | 0.0319     | 1.1447   | 1.1390         | 19.039  |

- **Interpretation**:
  - The mean volatility of ~0.0145 indicates relatively low variability in returns across stocks during the studied period.
  - The MA Ratio and Price-MA Ratio are centered around 1, reflecting a close alignment between stock prices and moving averages.
  - Volume (log-transformed) shows moderate variability, with certain stocks experiencing spikes in trading activity.

---

### **8.2 Correlation Analysis**

#### **8.2.1 Adjusted Close Prices**

![adj_close_corr_heatmap](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/EDA_Plots/adj_close_corr_heatmap.png)

- The correlation heatmap reveals strong positive relationships between the adjusted close prices of all stocks:
  - **Highest Correlations**:
    - Microsoft (MSFT) and Amazon (AMZN): \(0.95\)
    - Google (GOOGL) and Amazon (AMZN): \(0.95\)
  - **Lowest Correlation**:
    - Meta (META) and Apple (AAPL): \(0.83\)

- **Interpretation**:
  - These high correlations suggest that these technology stocks moved in tandem during the analyzed period, likely driven by common market factors like sector trends or macroeconomic conditions.

---

#### **8.2.2 Characteristics**

![characteristics_corr_heatma](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/EDA_Plots/characteristics_corr_heatmap.png)

- **Correlation Highlights**:
  - **Volatility and Volume**: Moderately positive correlation (\(0.48\)), suggesting that periods of high trading activity are associated with increased volatility.
  - **Volatility and Price-MA Ratio**: Strong negative correlation (\(-0.55\)), indicating that high volatility corresponds to prices diverging from their moving averages.
  - **MA Ratio and Other Characteristics**: Weak correlations (\(-0.27\) to \(0.04\)) suggest that MA Ratio captures a largely independent dimension of stock behavior.

- **Interpretation**:
  - These relationships provide insights into the interaction between market conditions (e.g., volatility) and pricing trends, highlighting key characteristics that may influence latent factors.

---

### **8.3 Visualizations**

#### **8.3.1 Pairplot of Characteristics**

![characteristics_pairplot](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/EDA_Plots/characteristics_pairplot.png)

- The pairplot shows scatter plots and distributions for each characteristic pair:
  - **Clustered Patterns**: Observed between Volatility and Volume, supporting their moderate positive correlation.
  - **Independent Spread**: MA Ratio exhibits a relatively uniform spread, consistent with its weak correlation with other characteristics.

- **Interpretation**:
  - The pairplot visually reinforces the statistical relationships and provides insights into how these characteristics interact.

---

### **8.4 Observed Trends**

#### **8.4.1 Volatility Over Time**

![volatility_over_time](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/EDA_Plots/volatility_over_time.png)

- Volatility appears cyclical, with periodic spikes likely reflecting broader market events or earnings announcements.

#### **8.4.2 Volume Over Time**

![volume_over_time](https://github.com/aditya-saxena-7/ipca/blob/main/real_data/EDA_Plots/volume_over_time.png)

- Volume trends are relatively stable but exhibit occasional surges, potentially linked to news events or significant market movements.

---

### **Conclusion**

The EDA highlights several critical aspects of the dataset:
- **Correlations**:
  - High correlations between adjusted close prices suggest a strong influence of common market-wide factors on these technology stocks.
  - Relationships between characteristics provide useful starting points for understanding latent factors influencing returns.
- **Key Characteristics**:
  - Volatility and Price-MA Ratio exhibit the most distinct relationships with other variables, suggesting their potential importance in latent factor modeling.
- **Robust Dataset**:
  - The dataset is well-structured, with consistent and interpretable relationships across variables, making it suitable for advanced modeling techniques like IPCA.
