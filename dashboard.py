"""
TSA Capstone Dashboard — run with:  streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="TSA Capstone 2026",
    page_icon="📈",
    layout="wide",
)

# ── colour palette ─────────────────────────────────────────────────────────────
COLORS = {
    'ARIMA'   : '#4C8BE8',
    'Prophet' : '#F5A623',
    'LSTM'    : '#27AE60',
    'Ensemble': '#8E44AD',
}

STOCKS = {
    'RELIANCE.NS'   : 'Reliance',
    'HDFCBANK.NS'   : 'HDFC Bank',
    'INFY.NS'       : 'Infosys',
    'SUNPHARMA.NS'  : 'Sun Pharma',
    'MARUTI.NS'     : 'Maruti',
    'HINDUNILVR.NS' : 'HUL',
    'TATASTEEL.NS'  : 'Tata Steel',
}

# ── data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_metrics():
    return pd.read_csv('results/03_model_metrics.csv')

@st.cache_data
def load_portfolio():
    return pd.read_csv('results/05_portfolio_allocation.csv')

@st.cache_data
def load_comparison():
    try:
        return pd.read_csv('results/08_prediction_vs_actual.csv')
    except FileNotFoundError:
        return None

@st.cache_data
def load_returns():
    try:
        return pd.read_csv('results/08_portfolio_return.csv')
    except FileNotFoundError:
        return None

@st.cache_data
def load_close():
    try:
        close = pd.read_csv('data/close_prices_clean.csv', index_col=0, parse_dates=True)
        return close
    except FileNotFoundError:
        return None

def load_5day():
    try:
        return pd.read_csv('results/03_5day_forecasts.csv')
    except FileNotFoundError:
        return None

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("TSA Capstone 2026 — NSE Stock Forecasting & Portfolio")
st.caption("Consulting & Analytics Club | IIT Guwahati | Tanishka Tyagi")
st.markdown("---")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Price History", "Model Comparison", "Portfolio Allocation",
     "5-Day Forecasts", "Trade Performance"]
)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.header("Project Overview")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Stocks", "7")
    col2.metric("Data Window", "2021 – 2025")
    col3.metric("Models", "ARIMA + Prophet")
    col4.metric("", "+ LSTM")
    col5.metric("Capital", "₹10,00,000")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Models & Test-Period Accuracy")
        df_m = load_metrics()
        df_m = df_m.groupby('model', as_index=False).mean(numeric_only=True).round(2)
        df_m = df_m.rename(columns={'model': 'Model'})
        df_m = df_m.sort_values('MAPE')
        st.dataframe(df_m[['Model', 'MAPE', 'RMSE', 'DirAcc']].reset_index(drop=True),
                     use_container_width=True)

    with col_b:
        st.subheader("Portfolio Summary")
        port = load_portfolio()
        if port is not None:
            total_deployed = port['Deployed (₹)'].sum()
            st.metric("Total Deployed", f"₹{total_deployed:,.0f}")
            st.metric("Cash Reserve", f"₹{1_000_000 - total_deployed:,.0f}")
            st.metric("Stocks in Portfolio", len(port))

    st.markdown("---")
    st.subheader("Trade Outcome (11–12 May 2026)")
    col_x, col_y, col_z = st.columns(3)
    col_x.metric("Portfolio Return", "−2.01%", delta="−2.01%")
    col_y.metric("Total P&L", "−₹19,837")
    col_z.metric("Directional Accuracy", "57.1%", delta="+7.1% vs random")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRICE HISTORY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Price History":
    st.header("Historical Price Data (2021–2025)")

    close = load_close()
    if close is None:
        st.warning("close_prices_clean.csv not found. Run notebook 1 first.")
    else:
        selected = st.multiselect(
            "Select stocks to display",
            options=list(STOCKS.values()),
            default=list(STOCKS.values())
        )
        sym_map = {v: k for k, v in STOCKS.items()}
        selected_syms = [sym_map[s] for s in selected if s in sym_map]

        if selected_syms:
            tab1, tab2 = st.tabs(["Raw Prices", "Base-100 Normalised"])

            with tab1:
                fig, ax = plt.subplots(figsize=(14, 5))
                for sym in selected_syms:
                    if sym in close.columns:
                        ax.plot(close.index, close[sym], label=STOCKS[sym], linewidth=1)
                ax.set_ylabel("Price (₹)")
                ax.set_title("Closing Prices")
                ax.legend(fontsize=8)
                st.pyplot(fig)
                plt.close()

            with tab2:
                fig, ax = plt.subplots(figsize=(14, 5))
                for sym in selected_syms:
                    if sym in close.columns:
                        series = close[sym].dropna()
                        normalised = series / series.iloc[0] * 100
                        ax.plot(normalised.index, normalised, label=STOCKS[sym], linewidth=1)
                ax.axhline(100, color='black', linestyle='--', linewidth=0.8)
                ax.set_ylabel("Normalised Price (Base = 100)")
                ax.set_title("Relative Performance (Jan 2021 = 100)")
                ax.legend(fontsize=8)
                st.pyplot(fig)
                plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.header("Model Comparison — Test Period (Jul–Dec 2025)")

    df_raw = load_metrics()
    df = df_raw.groupby(['stock', 'model'], as_index=False).mean(numeric_only=True)
    stocks = list(STOCKS.values())
    models = sorted(df['model'].unique())
    x = np.arange(len(stocks))
    w = 0.25

    metric = st.radio("Metric", ["MAPE (%)", "RMSE (₹)", "DirAcc (%)"], horizontal=True)
    col_map = {"MAPE (%)": "MAPE", "RMSE (₹)": "RMSE", "DirAcc (%)": "DirAcc"}
    m_col = col_map[metric]

    piv = df.pivot_table(index='stock', columns='model', values=m_col, aggfunc='mean').reindex(stocks).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, model in enumerate(models):
        vals = piv[model].values if model in piv.columns else [0]*len(stocks)
        ax.bar(x + i * w, vals, w, label=model,
               color=COLORS.get(model, '#888'), alpha=0.85)

    if m_col == "DirAcc":
        ax.axhline(50, color='red', linestyle='--', linewidth=1.2, label='Random (50%)')

    ax.set_xticks(x + w)
    ax.set_xticklabels(stocks, rotation=20, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by Model per Stock", fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Average Metrics by Model")
    avg = df.groupby('model')[['MAPE', 'RMSE', 'DirAcc']].mean().round(2)
    st.dataframe(avg, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PORTFOLIO ALLOCATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Portfolio Allocation":
    st.header("Portfolio Allocation — ₹10,00,000")

    port = load_portfolio()
    if port is None:
        st.warning("05_portfolio_allocation.csv not found. Run notebook 4/5 first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Allocation Breakdown")
            st.dataframe(
                port[['Stock', 'Weight (%)', 'Shares', 'Deployed (₹)']].reset_index(drop=True),
                use_container_width=True
            )

        with col2:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.pie(port['Weight (%)'], labels=port['Stock'], autopct='%1.1f%%',
                   startangle=90, pctdistance=0.82)
            ax.set_title("Portfolio Weight by Stock", fontweight='bold')
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.subheader("Strategy Rationale")
        st.markdown("""
| Strategy | Logic | Effect |
|----------|-------|--------|
| **A — Forecast-Guided** | Allocate proportionally to predicted positive return | High weight to HUL, Sun Pharma, HDFC Bank |
| **B — Inverse-Volatility** | 1/σ weighting — stable stocks get more capital | Pulls weight toward HDFC Bank, Reliance |
| **C — Inverse-Correlation** | Stocks less correlated with the portfolio get more | Promotes diversification |
| **D — Sector Momentum** | 30-day trailing return drives allocation | Rewards recent outperformers |
| **Final (A+B)** | Equal blend of A and B | Used for actual StockGro trades |
""")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — 5-DAY FORECASTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "5-Day Forecasts":
    st.header("5-Day Forecasts (ARIMA + Prophet Ensemble)")
    st.caption("Forecasts generated from data ending Dec 2025. Run from end-of-training price levels.")

    df5 = load_5day()
    if df5 is None:
        st.warning("results/03_5day_forecasts.csv not found. Run notebook 3 to the end first.")
    else:
        stock_filter = st.selectbox("Select stock", ["All"] + list(STOCKS.values()))

        if stock_filter != "All":
            df5 = df5[df5['Stock'] == stock_filter]

        pivot = df5.pivot_table(
            index=['Stock', 'Symbol'], columns='Day',
            values=['ARIMA_Forecast', 'Prophet_Forecast', 'Ensemble_Forecast']
        ).round(2)
        pivot.columns = [f'{v} D{d}' for v, d in pivot.columns]
        st.dataframe(pivot, use_container_width=True)

        if stock_filter != "All":
            fig, ax = plt.subplots(figsize=(10, 4))
            subset = df5.sort_values('Day')
            ax.plot(subset['Day'], subset['ARIMA_Forecast'],   marker='o', label='ARIMA',   color=COLORS['ARIMA'])
            ax.plot(subset['Day'], subset['Prophet_Forecast'], marker='s', label='Prophet', color=COLORS['Prophet'])
            ax.plot(subset['Day'], subset['Ensemble_Forecast'],marker='^', label='Ensemble',color=COLORS['Ensemble'],
                    linewidth=2, linestyle='--')
            ax.set_xlabel("Day")
            ax.set_ylabel("Forecast Price (₹)")
            ax.set_title(f"5-Day Forecast — {stock_filter}", fontweight='bold')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — TRADE PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Trade Performance":
    st.header("Trade Performance — 11–12 May 2026")

    ret = load_returns()
    cmp = load_comparison()

    if ret is None:
        st.warning("08_portfolio_return.csv not found. Run notebook 7/8 first.")
    else:
        col1, col2, col3 = st.columns(3)
        total_invested = ret['Invested (Rs)'].sum()
        total_value    = ret['Value D2 (Rs)'].sum()
        total_pnl      = total_value - total_invested
        port_ret       = total_pnl / total_invested * 100

        col1.metric("Total Invested", f"₹{total_invested:,.0f}")
        col2.metric("Portfolio Value (Day 2)", f"₹{total_value:,.0f}")
        col3.metric("Overall Return", f"{port_ret:+.2f}%",
                    delta=f"₹{total_pnl:+,.0f}")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Return % by Stock")
            colors_bar = ['#27AE60' if r >= 0 else '#E74C3C' for r in ret['Return (%)']]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(ret['Stock'], ret['Return (%)'], color=colors_bar, alpha=0.85)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_ylabel("Return (%)")
            ax.tick_params(axis='x', rotation=25)
            ax.set_title("Day 1 Buy → Day 2 Close", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.subheader("P&L by Stock (₹)")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(ret['Stock'], ret['P&L (Rs)'], color=colors_bar, alpha=0.85)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_ylabel("P&L (₹)")
            ax.tick_params(axis='x', rotation=25)
            ax.set_title("Absolute Profit & Loss", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")

        if cmp is not None:
            st.subheader("Predicted vs Actual (Ensemble Forecast, Day 1)")
            st.dataframe(
                cmp[['Stock', 'Pred D1 (Rs)', 'Actual D1 (Rs)', 'MAPE D1 (%)',
                      'Dir Predicted', 'Dir Actual', 'Dir Correct']].reset_index(drop=True),
                use_container_width=True
            )

            correct   = (cmp['Dir Correct'] == 'YES').sum()
            total_all = len(cmp)
            st.info(f"Directional Accuracy: {correct}/{total_all} stocks correct ({correct/total_all*100:.1f}%)")
