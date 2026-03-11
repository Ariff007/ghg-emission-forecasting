# 🌍 Carbon Emission Forecasting — Malaysia

**Author:** Ariff Azahari  
**Datasets:** `ghg_emissions.csv` · `air_pollution.csv`  
**Objective:** Forecast total GHG emissions and air pollutant concentrations for the next 5–10 years using multiple models.

---

## Pipeline Overview
1. Data Loading & Overview
2. Exploratory Data Analysis (EDA)
3. Stationarity Tests (ADF & KPSS)
4. Feature Engineering (Lag Features, Rolling Stats)
5. Model Training — Prophet + XGBoost + SARIMA
6. Model Evaluation (MAE, RMSE, MAPE)
7. Forecast Visualisation with Confidence Intervals

---

## 📦 Install & Import Dependencies

```python
# Install required libraries (run once)
%pip install pandas numpy matplotlib seaborn statsmodels scikit-learn xgboost prophet
```

```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import xgboost as xgb
from prophet import Prophet

# Plotting style
plt.style.use('dark_background')
sns.set_palette('muted')
ACCENT = '#4279a4'
ACCENT2 = '#ddd1ca'
FIG_W, FIG_H = 14, 5

print('✅ All libraries imported successfully')
```
---
## 🏁 Summary & Key Takeaways

| Aspect | Finding |
|---|---|
| **GHG Total Trend** | Consistent upward trend from 2014–2019, slight dip in 2020 (COVID-19), recovery in 2021 |
| **Dominant Sector** | Energy accounts for ~80% of total emissions |
| **LULUCF (forests)** | Acts as a carbon sink (negative values), but shrinking |
| **Best Model (GHG)** | **Prophet** — better uncertainty quantification with very small annual dataset |
| **Best Model (Air)** | **XGBoost** — lower MAPE on test set with sufficient monthly data |
| **CO Trend** | Downward trend post-2017, with a clear dip in 2020 (lockdowns) |
| **PM10/PM2.5** | Seasonal spikes in mid-year (haze season) visible |

### Recommendations
- Acquire post-2021 GHG data for more robust modelling and train with **SARIMA** or **LSTM** when ≥15 annual points are available.
- Use air pollution forecasts as a **proxy indicator** for tracking progress toward carbon neutrality targets.
- Consider **exogenous variables** (GDP growth, energy mix, policy interventions) to improve forecast accuracy using SARIMAX or Prophet regressors.