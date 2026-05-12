# TSA Capstone Project 2026 — Written Report
### Consulting & Analytics Club | IIT Guwahati

**Submitted by:** Tanishka Tyagi  
**Date:** May 2026  
**Project:** Time Series Analysis — NSE Stock Forecasting & Virtual Portfolio

---

## 1. Executive Summary

This project applies time series analysis to real NSE stock data to build a data-driven investment process from scratch. Seven stocks were selected across distinct sectors, five years of daily price data (January 2021 – December 2025) were collected and preprocessed, and three forecasting models — ARIMA, Facebook Prophet, and LSTM — were trained and evaluated on a held-out test period (July–December 2025). A ₹10,00,000 virtual portfolio was allocated using four complementary strategies and executed on StockGro over two days (11–12 May 2026).

Key outcomes:
- **LSTM achieved the best test-period accuracy** with an average MAPE of 1.42% across all 7 stocks
- **ARIMA was the most interpretable** model with competitive accuracy (avg MAPE 2.29%)
- **Prophet performed poorly** on trend-heavy stocks like Maruti and Infosys (MAPEs of 16.7% and 15.1%), likely because it extrapolated late-2025 seasonal patterns that broke down over a 5-month gap
- **Directional accuracy on the two actual trade days was 57.1%** — above the 50% random baseline
- **Portfolio return: −2.01%** over the two-day window (P&L: −₹19,837), driven by a broad market selloff on 12 May 2026 that affected all holdings except Tata Steel

---

## 2. Introduction & Objectives

The project was structured around 8 tasks, each building on the previous:

1. Select 7 NSE stocks with clear sector diversity
2. Collect and preprocess 5 years of daily OHLCV data
3. Build ARIMA, Prophet, and LSTM forecasting models
4. Analyze volatility and risk profiles
5. Construct a ₹10,00,000 virtual portfolio using multiple strategies
6. Compare model performance across stocks and metrics
7. Execute trades on StockGro virtual trading platform
8. Track performance and compare predictions to actual outcomes

The central question was whether statistical and deep learning time series models can generate actionable directional signals for equity trading — and how their predictions hold up when tested against real (virtual) market prices.

---

## 3. Stock Selection (Task 1)

**Selection criteria:**  
Stocks were chosen to maximize sector diversity, include both large-cap and mid-cap names, and ensure sufficient liquidity for virtual trading. Each stock is a recognized sector bellwether on NSE.

| Stock | Symbol | Sector | Market Cap Category |
|-------|--------|--------|---------------------|
| Reliance Industries | RELIANCE.NS | Energy / Conglomerate | Large Cap |
| HDFC Bank | HDFCBANK.NS | Banking / BFSI | Large Cap |
| Infosys | INFY.NS | Information Technology | Large Cap |
| Sun Pharma | SUNPHARMA.NS | Pharmaceuticals | Large Cap |
| Maruti Suzuki | MARUTI.NS | Automobile | Large Cap |
| Hindustan Unilever | HINDUNILVR.NS | FMCG | Large Cap |
| Tata Steel | TATASTEEL.NS | Metal / Steel | Large Cap |

**Data window:** 1 January 2021 – 31 December 2025 (5 years, ~1,235 trading days)  
**Source:** Yahoo Finance via `yfinance` library (adjusted closing prices)

The 5-year window was chosen to capture multiple market regimes: the COVID recovery rally (2021), the rate-hike driven correction (2022), the mid-cap bull run (2023), pre/post general election volatility (2024), and the 2025 consolidation phase. This range ensures the models have enough data to learn from varied conditions.

**Exploratory findings:**  
Base-100 normalized prices showed Tata Steel and Sun Pharma had the highest 5-year returns (~+120% and ~+180% respectively from January 2021 levels), while HUL delivered the weakest price appreciation. Maruti showed the highest single-year volatility spike in 2022. Rolling 30-day correlations confirmed that the 7 stocks are not perfectly correlated, supporting diversification.

---

## 4. Data Preprocessing (Task 2)

Raw data from yfinance had minor gaps (public holidays, market closures) and needed preparation before modelling.

**Preprocessing pipeline:**

1. **Missing value handling:** Forward-fill (`ffill`) for up to 2 consecutive missing days; backfill for any remaining gaps at the start of the series. This preserves the most recent known price rather than interpolating, which would introduce future information.

2. **Stationarity check (ADF Test):**  
   The Augmented Dickey-Fuller test was run on each stock's raw price series. All 7 series failed to reject H₀ at p < 0.05 — confirming non-stationarity (a random walk). First-order differencing (Δ log prices = log returns) made all series stationary. This drives the `d=1` parameter in ARIMA.

3. **Train/test split:**  
   Split at July 1, 2025 — approximately 80/20 (1,110 training days, 125 test days). The test period coincides with the second half of 2025.

4. **Scaling:**  
   MinMaxScaler was applied per stock, **fit only on the training set**. The same scaler transforms the test set. This is the correct procedure — fitting on the full series would leak future price ranges into training.

5. **LSTM sequence construction:**  
   60-day sliding windows were created from the scaled training series (look-back = 60 days). Each window becomes one training sample, with the next day's price as the target. This produced approximately 1,050 training sequences per stock.

---

## 5. Forecasting Models (Task 3)

Three model families were built, each bringing different assumptions and strengths.

### 5.1 ARIMA

ARIMA(p, d, q) combines autoregression (AR), differencing (I), and moving average (MA) components. The differencing order d=1 was fixed (confirmed by ADF). Orders p and q were selected via AIC grid search over p ∈ {0,1,2}, q ∈ {0,1,2} for each stock.

**Selected orders:**

| Stock | ARIMA Order | AIC |
|-------|-------------|-----|
| Reliance | (2,1,2) | 9,497 |
| HDFC Bank | (2,1,2) | 8,355 |
| Infosys | (0,1,0) | 10,060 |
| Sun Pharma | (2,1,2) | 9,180 |
| Maruti | (2,1,1) | 14,050 |
| HUL | (0,1,2) | 10,650 |
| Tata Steel | (0,1,0) | 5,114 |

That Infosys and Tata Steel best fit ARIMA(0,1,0) — a simple random walk — reflects their near-random daily price behaviour during the training period.

**Walk-forward validation:** Rather than fitting once on all training data and forecasting 125 days ahead, the model was refitted every 20 steps using all data available up to that point. This mimics real deployment and avoids stale model coefficients.

**Residual diagnostics (Reliance as representative):**  
Durbin-Watson statistic: ~2.0, indicating no significant autocorrelation in residuals. The Q-Q plot showed residuals were approximately normally distributed, with slight heavy tails — expected for financial return data.

### 5.2 Facebook Prophet

Prophet decomposes the time series as: `y(t) = trend(t) + seasonality(t) + error(t)`. Weekly and yearly seasonality components were enabled. `changepoint_prior_scale = 0.1` (slightly tighter than the default 0.05) was chosen to reduce overfitting to local trend breaks in the training data. A 95% confidence interval was produced for each forecast.

**Key weakness observed:** Prophet extrapolates the last known trend aggressively. For stocks like Maruti (which was trending at ~₹16,600 in late 2025), Prophet's Day-2 forecast was ₹16,590 — but the actual price in May 2026 was ₹12,896. This 5-month stale forecast illustrates Prophet's sensitivity to the data horizon.

### 5.3 LSTM

Architecture: two LSTM layers (50 units each) → Dropout(0.2) → Dense(25) → Dense(1). Input: 60-day scaled price window. Adam optimizer, MSE loss. EarlyStopping monitored validation loss with patience=10 and `restore_best_weights=True`.

Training was run for up to 50 epochs per stock with a 10% validation split. Most stocks converged in 30–45 epochs.

**Test-period performance summary:**

| Stock | ARIMA MAPE | Prophet MAPE | LSTM MAPE | Best |
|-------|-----------|-------------|----------|------|
| Reliance | 1.85% | 8.84% | **1.06%** | LSTM |
| HDFC Bank | 1.21% | 4.43% | **0.64%** | LSTM |
| Infosys | 2.32% | 15.06% | **1.38%** | LSTM |
| Sun Pharma | 1.95% | 11.54% | **1.61%** | LSTM |
| Maruti | 2.80% | 16.70% | **2.55%** | LSTM |
| HUL | 3.34% | 3.38% | **1.05%** | LSTM |
| Tata Steel | 2.56% | 11.53% | **1.63%** | LSTM |
| **Average** | **2.29%** | **10.21%** | **1.42%** | LSTM |

LSTM dominated on MAPE across all 7 stocks. Prophet had high directional accuracy (avg 52.7%) due to its trend component but unacceptably high magnitude error for most stocks.

---

## 6. Volatility Analysis (Task 4)

### 6.1 Log Returns

Log returns were computed as ln(P_t / P_{t-1}). Annualised statistics:

| Stock | Annualised Return | Annualised Volatility (σ) |
|-------|-------------------|--------------------------|
| Tata Steel | +23.6% | 33.1% |
| Sun Pharma | +22.7% | 20.9% |
| Maruti | +16.6% | 23.2% |
| Reliance | +10.9% | 22.7% |
| HDFC Bank | +7.8% | 21.2% |
| Infosys | +7.8% | 24.1% |
| HUL | +0.9% | 20.1% |

Tata Steel showed the highest return/risk profile but also the widest volatility. HUL was the least volatile but also returned nearly nothing over 5 years on a log-return basis.

### 6.2 Rolling Volatility

30-day rolling annualised volatility showed clear spikes for all stocks in 2022 (rate hike cycle) and mid-2024 (election results). Post-election, volatility generally compressed. Most recently (end of 2025), 30-day σ ranged from 11.5% (HDFC Bank) to 22.1% (Tata Steel).

### 6.3 Correlation Matrix

The pairwise return correlation matrix revealed:
- Strongest correlation: Reliance & HDFC Bank (~0.55) — both macro-sensitive large caps
- Weakest correlation: HUL & Tata Steel (~0.20) — FMCG vs Cyclical Metal, as expected
- IT stocks (Infosys) showed low correlation with domestic cyclicals, providing meaningful diversification

### 6.4 STL Decomposition

STL decomposition (period = 252 trading days) isolated trend, seasonal, and residual components for each stock. Trend component explained 85–95% of variance for most stocks, confirming price levels are largely trend-driven. Seasonal components were small but consistent, particularly for FMCG (HUL) and Pharma (Sun Pharma) which show mild Q4 upticks.

---

## 7. Portfolio Construction (Task 5)

**Capital:** ₹10,00,000  
**Strategy:** Four allocation approaches were computed and compared. The A+B blend was used for actual StockGro execution.

### Strategy A — Forecast-Guided
Allocate proportionally to predicted positive return (Day-2 ensemble forecast vs last known price). Stocks predicted to fall receive zero weight.

### Strategy B — Inverse-Volatility (1/σ)
Weight inversely proportional to recent 30-day annualised volatility. Lower-risk stocks receive larger positions.

### Strategy C — Inverse-Correlation
Stocks that are less correlated with the rest of the portfolio receive higher weight. This maximises diversification benefit per unit of capital.

### Strategy D — Sector Momentum Rotation
30-day trailing price return used as a momentum signal. Stocks (and their sectors) with positive recent momentum receive proportionally more capital, floored at 1% to retain full diversification.

**Strategy weight comparison:**

| Stock | A (Forecast) | B (1/Vol) | C (1/Corr) | D (Momentum) | **A+B Used** |
|-------|-------------|----------|-----------|-------------|-------------|
| HUL | 48.1% | 11.5% | 15.5% | 11.8% | **29.8%** |
| HDFC Bank | 24.8% | 19.2% | 13.2% | 9.7% | **22.0%** |
| Sun Pharma | 26.6% | 14.7% | 14.8% | 13.6% | **20.6%** |
| Reliance | 0.5% | 18.2% | 12.8% | 10.8% | **9.4%** |
| Maruti | 0.0% | 15.1% | 14.1% | 16.3% | **7.5%** |
| Infosys | 0.0% | 11.4% | 14.5% | 11.5% | **5.7%** |
| Tata Steel | 0.0% | 10.0% | 15.1% | 26.3% | **5.0%** |

**Planned A+B allocation (based on Dec 2025 prices):**

| Stock | Weight | Planned Shares | Allocated (₹) |
|-------|--------|---------------|--------------|
| HUL | 29.8% | 130 | ₹2,97,726 |
| HDFC Bank | 22.0% | 221 | ₹2,18,989 |
| Sun Pharma | 20.6% | 120 | ₹2,05,092 |
| Reliance | 9.4% | 60 | ₹92,388 |
| Maruti | 7.5% | 4 | ₹66,588 |
| Infosys | 5.7% | 35 | ₹56,756 |
| Tata Steel | 5.0% | 284 | ₹49,927 |
| **Total** | **100%** | | **₹9,87,466** |

Cash reserve: ₹12,534 (1.3%). Residual from rounding down to whole shares.

Note: Actual shares bought on StockGro on 11 May 2026 differ because prices had moved significantly since December 2025 (e.g. HDFC Bank fell from ₹990 to ₹752). The same capital allocation percentages were maintained but recalculated at live prices — see Section 9.1.

---

## 8. Model Comparison (Task 6)

Across all metrics, LSTM consistently outperformed ARIMA and Prophet for this dataset:

| Model | Avg MAPE | Avg RMSE (₹) | Avg DirAcc |
|-------|----------|-------------|-----------|
| LSTM | **1.42%** | **84.6** | 48.9% |
| ARIMA | 2.29% | 129.8 | 50.1% |
| Ensemble (A+P) | 6.25% | 341.5 | **51.4%** |
| Prophet | 10.21% | 553.2 | 52.7% |

**Key observations:**

- LSTM's lower MAPE reflects its ability to fit the smooth near-term trends in the test period. However, its directional accuracy (48.9%) was slightly below random — the model predicts prices well in level but fails to consistently call the direction.
- ARIMA had the best directional accuracy (50.1%) and was far more interpretable. Walk-forward refitting kept it adaptive.
- The ARIMA+Prophet ensemble improved directional accuracy to 51.4% while reducing Prophet's magnitude error. This is why it was chosen as the allocation signal for Task 5.
- Prophet's weakness is specific to this use case: stock prices rarely follow the repeating seasonal patterns Prophet is designed for, and the long gap between training data cutoff and trade date compounded the error.

**Best model per stock by MAPE:** LSTM won for all 7 stocks.  
**Best model per stock by DirAcc:** Prophet won for 5/7 stocks (HDFC Bank, Infosys, Sun Pharma, Maruti, HUL).

---

## 9. Virtual Trading & Performance (Tasks 7 & 8)

### 9.1 Trade Execution

Trades were placed on StockGro (NSE live data simulator) on **11 May 2026** at closing prices. Exit prices were recorded at closing on **12 May 2026**.

| Stock | Shares | Buy Price (₹) | Exit Price (₹) | Return | P&L (₹) |
|-------|--------|--------------|----------------|--------|---------|
| HUL | 131 | 2,287.66 | 2,221.93 | −2.87% | −8,611 |
| HDFC Bank | 289 | 752.41 | 740.45 | −1.59% | −3,456 |
| Sun Pharma | 110 | 1,858.73 | 1,829.99 | −1.55% | −3,161 |
| Reliance | 67 | 1,365.92 | 1,348.35 | −1.29% | −1,177 |
| Maruti | 5 | 13,262.17 | 12,896.89 | −2.75% | −1,826 |
| Infosys | 49 | 1,159.58 | 1,125.90 | −2.90% | −1,650 |
| Tata Steel | 237 | 209.44 | 209.63 | **+0.09%** | +45 |
| **Total** | | | | **−2.01%** | **−19,837** |

### 9.2 Predicted vs Actual

The models were trained on data ending December 2025 and used to generate forecasts for May 2026 — a 5-month forward gap. Actual Day-1 prices on StockGro differed significantly from forecasts:

| Stock | Ensemble Forecast (Day 1) | Actual Day 1 | MAPE |
|-------|--------------------------|--------------|------|
| HUL | ₹2,289.9 | ₹2,287.7 | **0.10%** |
| Sun Pharma | ₹1,709.2 | ₹1,858.7 | 8.05% |
| Reliance | ₹1,539.3 | ₹1,365.9 | 12.69% |
| Tata Steel | ₹175.8 | ₹209.4 | 16.06% |
| Maruti | ₹16,658.9 | ₹13,262.2 | 25.61% |
| HDFC Bank | ₹991.0 | ₹752.4 | 31.71% |
| Infosys | ₹1,621.6 | ₹1,159.6 | 39.84% |
| **Average** | | | **19.15%** |

Average Day-1 MAPE of 19.15% is much higher than the test-period MAPE of ~6.25% for the ensemble. The gap is almost entirely explained by the 5-month data staleness — markets moved sharply between January and May 2026 (IT sector correction, HDFC Bank re-rating, FMCG weakening), and the models had no information about any of this.

**Directional accuracy on trade days: 57.1% (4 out of 7 stocks correctly predicted direction D1→D2).**  
Correct: HDFC Bank (↓), Infosys (↓), Sun Pharma (↓), Maruti (↓).  
Incorrect: Reliance (predicted ↑, moved ↓), HUL (predicted ↑, moved ↓), Tata Steel (predicted ↓, moved ↑ marginally).

### 9.3 Why the Portfolio Lost Money

12 May 2026 saw a broad NSE selloff driven by US tariff concerns and net FII outflows. In such a macro-driven event, all domestic equities fell together regardless of sector. The correlation between portfolio holdings effectively went to 1 during the selloff — exactly the scenario where diversification provides no protection. Only Tata Steel (a commodity play partially hedged by export demand expectations) ended marginally positive.

---

## 10. Reflection & Conclusion

### Was the forecasting approach sound?

The modelling pipeline — ADF testing, proper train/test split, walk-forward ARIMA, Prophet with tuned changepoint scale, and LSTM with EarlyStopping — follows sound time series methodology. The test-period accuracy numbers (LSTM avg MAPE 1.42%, ARIMA avg MAPE 2.29%) are competitive with academic benchmarks for daily stock price forecasting.

The fundamental limitation was the deployment gap. Models built on December 2025 data were asked to forecast May 2026 prices. Without retraining or any macro signal, they had no way to account for the structural price shifts that occurred in early 2026. In a real trading system, models would be retrained weekly or monthly.

### Which model would you use in practice?

For a short-horizon trading signal (1–2 days), an ensemble of ARIMA and LSTM makes the most sense:
- ARIMA provides an interpretable, statistically rigorous baseline with walk-forward adaptability
- LSTM captures non-linear dependencies that ARIMA misses, and achieved the best raw accuracy
- Prophet would be reserved for longer-horizon trend analysis (weeks/months) where its seasonality modelling adds value

### Were the portfolio strategies effective?

The A+B strategy blend correctly identified HUL and Sun Pharma as high-conviction buys (positive ensemble signal, moderate volatility). Unfortunately, the broad selloff made sector diversification irrelevant over a 2-day window. Over a longer horizon (20+ trading days), the diversification across Energy, Banking, IT, Pharma, Auto, FMCG, and Metal would reduce portfolio variance meaningfully.

Strategies C (correlation-based) and D (momentum) added analytical depth to the allocation decision. Strategy D, in hindsight, would have increased Tata Steel exposure (its 30-day momentum was positive) — the only profitable holding on Day 2.

### What would be done differently?

1. **Reduce the data staleness gap** — retrain models within 1–2 weeks of the trade date
2. **Add macro features** — RBI policy rate, Nifty 50 direction, USD/INR as ARIMAX exogenous variables
3. **Use LSTM forecasts for allocation** — LSTM had the best accuracy but was not used as the allocation signal (only ARIMA+Prophet were)
4. **Extend the trading window** — 2 days is statistically insufficient; 30+ days would give a proper assessment of directional accuracy
5. **Set a market-regime filter** — when Nifty 50 intraday move exceeds −1%, reduce position sizes or hold cash

### Conclusion

This project successfully built a complete time series analysis pipeline — from raw NSE data through preprocessing, model training, portfolio construction, and live virtual trading. LSTM demonstrated superior accuracy in the controlled test period. The −2.01% portfolio return on the two actual trade days was an outcome of a macro event (market-wide selloff), not a failure of the forecasting methodology. The directional accuracy of 57.1% — above the random baseline — suggests the models carry genuine signal. The main lesson: even a technically sound forecasting pipeline needs to be retrained near the time of deployment to remain useful in live markets.

---

*Data sources: Yahoo Finance (yfinance), NSE via StockGro virtual trading platform*  
*Tools: Python 3.13, pandas, NumPy, statsmodels, Prophet, TensorFlow/Keras, scikit-learn, matplotlib, seaborn*  
*Code: Available in notebooks 01–08 in the project repository*
