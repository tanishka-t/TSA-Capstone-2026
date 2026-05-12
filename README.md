# TSA Capstone Project 2026 — Time Series Analysis
## Consulting & Analytics Club | IIT Guwahati
**Submitted by:** Tanishka Tyagi

---

## Project Summary

End-to-end time series analysis pipeline on 7 NSE stocks:
- 5 years of daily data (Jan 2021 – Dec 2025) collected and preprocessed
- Three forecasting models built and evaluated: ARIMA, Facebook Prophet, LSTM
- ₹10,00,000 virtual portfolio allocated using 4 strategies and traded on StockGro
- Actual trades executed: **11–12 May 2026** | Portfolio return: **−2.01%**

---

## Folder Structure

```
TSA Project/
│
├── 01_stock_selection.ipynb          ← Task 1: stock picks, EDA, trend summary
├── 02_preprocessing.ipynb            ← Task 2: ffill, ADF test, scaling, sequences
├── 03_forecasting.ipynb              ← Task 3: ARIMA, Prophet, LSTM + 5-day forecasts
├── 04_05_volatility_portfolio.ipynb  ← Tasks 4+5: volatility analysis + portfolio construction
├── 06_model_comparison.ipynb         ← Task 6: MAPE/RMSE/DirAcc charts + ensemble eval
├── 07_08_performance_tracking.ipynb  ← Tasks 7+8: StockGro trades + predicted vs actual
│
├── dashboard.py                      ← Streamlit interactive dashboard
├── report.md                         ← 10-section written report (submit this)
├── requirements.txt                  ← Python dependencies
│
├── data/
│   ├── close_prices_raw.csv          ← raw yfinance download
│   ├── close_prices_clean.csv        ← after forward-fill
│   ├── train_data.csv                ← Jan 2021 – Jun 2025 (1,110 days)
│   ├── test_data.csv                 ← Jul 2025 – Dec 2025 (125 days)
│   ├── scalers.pkl                   ← per-stock MinMaxScalers (fit on train only)
│   ├── arima_forecasts.pkl           ← ARIMA 5-day forecasts
│   └── prophet_forecasts.pkl         ← Prophet 5-day forecasts
│   └── lstm_sequences.pkl            ← 60-day (X, y) windows for LSTM
│
├── models/
│   └── lstm_*.keras                  ← saved LSTM model per stock
│
└── results/
    ├── 01_trend_summary.csv
    ├── 02_adf_results.csv
    ├── 03_model_metrics.csv          ← MAPE / RMSE / DirAcc for all models
    ├── 03_5day_forecasts.csv         ← ARIMA + Prophet + Ensemble Days 1–5
    ├── 03_arima_residuals.png
    ├── 04_rolling_volatility.png
    ├── 04_correlation_matrix.png / .csv
    ├── 04_stl_decomposition.png
    ├── 05_portfolio_allocation.csv   ← final weight / shares / deployed ₹
    ├── 06_model_comparison_pivot.csv
    ├── 06_ensemble_metrics.csv
    ├── 08_prediction_vs_actual.csv
    └── 08_portfolio_return.csv
```

---

## Setup

### 1. Create virtual environment and install packages
```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install streamlit           # for dashboard (not in requirements.txt)
```

### 2. Register kernel for Jupyter / VS Code
```bash
.\.venv\Scripts\python.exe -m ipykernel install --user --name tsaproject --display-name "TSA Project"
```

### 3. Run notebooks in order
Open in VS Code or Jupyter and run top-to-bottom in sequence:
```
01 → 02 → 03 → 04_05 → 06 → 07_08
```
Each notebook saves outputs to `data/`, `models/`, and `results/` for the next notebook to load.

### 4. Launch dashboard
```bash
.\.venv\Scripts\streamlit.exe run dashboard.py
```

---

## Stocks

| Symbol | Company | Sector |
|--------|---------|--------|
| RELIANCE.NS | Reliance Industries | Energy / Conglomerate |
| HDFCBANK.NS | HDFC Bank | Banking |
| INFY.NS | Infosys | Information Technology |
| SUNPHARMA.NS | Sun Pharmaceutical | Pharmaceuticals |
| MARUTI.NS | Maruti Suzuki | Automobile |
| HINDUNILVR.NS | Hindustan Unilever | FMCG |
| TATASTEEL.NS | Tata Steel | Metal / Steel |

---

## Models

| Model | Avg MAPE (test) | Avg DirAcc | Notes |
|-------|----------------|-----------|-------|
| LSTM | **1.42%** | 48.9% | Best accuracy; 2-layer 50-unit, EarlyStopping |
| ARIMA | 2.29% | 50.1% | AIC grid search; walk-forward refit every 20 steps |
| ARIMA+Prophet Ensemble | ~6.25% | 51.4% | Used for portfolio allocation signal |
| Prophet | 10.21% | 52.7% | Weak on trend-heavy stocks; high MAPE on Maruti/Infosys |

Test period: Jul–Dec 2025 (125 days). Train: Jan 2021–Jun 2025 (1,110 days).

---

## Portfolio Strategies

| Strategy | Logic | Weight Effect |
|----------|-------|--------------|
| **A — Forecast-Guided** | Proportional to predicted positive return | Heavy on HUL, Sun Pharma, HDFC Bank |
| **B — Inverse-Volatility** | 1/σ weighting | Pulls weight toward stable stocks (HDFC Bank, Reliance) |
| **C — Inverse-Correlation** | 1/avg_abs_ρ — less correlated = more weight | Promotes diversification |
| **D — Sector Momentum** | 30-day trailing return per stock, floor 1% | Rewards recent outperformers |
| **Final (A+B used for trades)** | Equal blend of A and B | Balanced return-risk signal |

**Final allocation used on StockGro (11 May 2026):**

| Stock | Weight | Shares Bought | Buy Price |
|-------|--------|--------------|-----------|
| HUL | 29.8% | 131 | ₹2,287.66 |
| HDFC Bank | 22.0% | 289 | ₹752.41 |
| Sun Pharma | 20.6% | 110 | ₹1,858.73 |
| Reliance | 9.4% | 67 | ₹1,365.92 |
| Maruti | 7.5% | 5 | ₹13,262.17 |
| Infosys | 5.7% | 49 | ₹1,159.58 |
| Tata Steel | 5.0% | 237 | ₹209.44 |

---

## Results

**Trade dates:** 11–12 May 2026 (StockGro virtual platform)

| Stock | Return | P&L |
|-------|--------|-----|
| Tata Steel | +0.09% | +₹45 |
| Reliance | −1.29% | −₹1,177 |
| HDFC Bank | −1.59% | −₹3,456 |
| Sun Pharma | −1.55% | −₹3,161 |
| Maruti | −2.75% | −₹1,826 |
| HUL | −2.87% | −₹8,611 |
| Infosys | −2.90% | −₹1,650 |
| **Portfolio** | **−2.01%** | **−₹19,837** |

Directional accuracy on trade days: **57.1%** (4/7 stocks, above 50% random baseline).  
Broad market selloff on 12 May 2026 (FII outflows + US tariff concerns) drove all holdings negative.

---

## Key Files for Submission

| File | Purpose |
|------|---------|
| `report.md` | 10-section written report with all findings and reflection |
| `01_stock_selection.ipynb` | Task 1 |
| `02_preprocessing.ipynb` | Task 2 |
| `03_forecasting.ipynb` | Task 3 |
| `04_05_volatility_portfolio.ipynb` | Tasks 4 & 5 |
| `06_model_comparison.ipynb` | Task 6 |
| `07_08_performance_tracking.ipynb` | Tasks 7 & 8 |
| `dashboard.py` | Optional visual dashboard |
