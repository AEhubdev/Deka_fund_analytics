import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.data_loader import load_and_preprocess_data
from src.metrics_calculator import compute_all_metrics

st.set_page_config(
    page_title="Deka-Deutschland Nebenwerte CF Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Deka-Deutschland Nebenwerte CF (LU0923076540)")
st.markdown("**Fund Performance & Risk Analysis Dashboard** *(Data span: 30.12.2025 – 30.09.2026)*")

# Sidebar Controls (Risk-free rate configuration only)
st.sidebar.header("Model Settings")
risk_free_rate = st.sidebar.number_input("Risk Free Rate (p.a.)", value=0.02, step=0.005, format="%.3f")

file_source = "data/Trainee-Aufgabe_2026.xlsx"

try:
    # Load and process data
    df, date_col, price_col = load_and_preprocess_data(file_source)
    metrics, drawdowns = compute_all_metrics(df, price_col=price_col, risk_free_rate=risk_free_rate)

    # KPI Row 1: Performance
    st.subheader("📊 Core Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cumulative Return", f"{metrics['Cumulative Return']:.2%}")
    col2.metric("Annualized Return (CAGR)", f"{metrics['Annualized Return (CAGR)']:.2%}")
    col3.metric("Start Price", f"{metrics['Start NAV / Price']:.2f} EUR")
    col4.metric("End Price", f"{metrics['End NAV / Price']:.2f} EUR")

    # KPI Row 2: Risk-Adjusted Ratios & Risk
    st.subheader("🛡️ Risk & Risk-Adjusted Measures")
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio (rf=2%)']:.2f}")
    col6.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}")
    col7.metric("Calmar Ratio", f"{metrics['Calmar Ratio']:.2f}")
    col8.metric("Max Drawdown", f"{metrics['Maximum Drawdown']:.2%}")

    st.markdown("---")

    # Tabs Section for Detailed Exploration
    tab1, tab2, tab3 = st.tabs(
        ["Price & Drawdown Analysis", "Return Distribution & Tail Risk", "Complete Metrics Inventory"])

    with tab1:
        fig_price = px.line(df, x=date_col, y=price_col, title="Fund NAV / Price History (30.12.2025 – 30.09.2026)")
        st.plotly_chart(fig_price, use_container_width=True)

        fig_dd = px.area(df, x=date_col, y=drawdowns, title="Under-water Drawdown Chart",
                         labels={'y': 'Drawdown Level'})
        fig_dd.update_traces(fillcolor="rgba(238, 75, 43, 0.3)", line_color="#EE4B2B")
        st.plotly_chart(fig_dd, use_container_width=True)

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_hist = px.histogram(df.dropna(), x='Daily_Return', nbins=40, title="Daily Return Distribution",
                                    marginal="box")
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_b:
            st.markdown("### Tail Risk & Distribution Insights")
            st.markdown(
                f"* **Daily Volatility:** `{metrics['Annualized Volatility'] / np.sqrt(252):.2%}` (Annualized: `{metrics['Annualized Volatility']:.2%}`)")
            st.markdown(f"* **Historical VaR (95% Daily):** `{metrics['Historical VaR (95% Daily)']:.2%}`")
            st.markdown(
                f"* **Expected Shortfall / CVaR (95%):** `{metrics['Historical Expected Shortfall / CVaR (95%)']:.2%}`")
            st.markdown(
                f"* **Best Day:** `{metrics['Best Daily Return']:.2%}` | **Worst Day:** `{metrics['Worst Daily Return']:.2%}`")
            st.markdown(f"* **Positive Trading Days:** `{metrics['Positive Days Ratio']:.1%}`")
            st.markdown(
                f"* **Skewness:** `{metrics['Return Skewness']:.2f}` | **Kurtosis:** `{metrics['Return Kurtosis']:.2f}`")

    with tab3:
        st.subheader("All Computed Quantitative Metrics")
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Indicator / Metric", "Value"])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Error processing the file: {e}")