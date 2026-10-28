import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_and_preprocess_data
from src.metrics_calculator import compute_all_metrics

st.set_page_config(
    page_title="Deka-Deutschland Nebenwerte CF Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Deka-Deutschland Nebenwerte CF (LU0923076540)")
st.markdown("**Fund Performance & Risk Analysis Dashboard (30.12.2025 – 30.09.2026)**")

# Sidebar Controls
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Excel File (.xlsx)", type=["xlsx", "xls"])
risk_free_rate = st.sidebar.number_input("Risk Free Rate (p.a.)", value=0.02, step=0.005, format="%.3f")

# Data Source selection: defaults to your specific file packaged in data/
if uploaded_file is not None:
    file_source = uploaded_file
else:
    file_source = "data/Trainee-Aufgabe_2026.xlsx"
    st.sidebar.info("Using default file: `data/Trainee-Aufgabe_2026.xlsx`.")

try:
    # Load and process data
    df, date_col, price_col = load_and_preprocess_data(file_source)
    metrics, drawdowns = compute_all_metrics(df, price_col=price_col, risk_free_rate=risk_free_rate)

    # KPI Row 1
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cumulative Return", f"{metrics['Cumulative Return']:.2%}")
    col2.metric("Annualized Return", f"{metrics['Annualized Return']:.2%}")
    col3.metric("Annualized Volatility", f"{metrics['Annualized Volatility']:.2%}")
    col4.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")

    # KPI Row 2
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
    col6.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}")
    col7.metric("Daily VaR (95%)", f"{metrics['Historical VaR (95%)']:.2%}")
    col8.metric("Daily CVaR (95%)", f"{metrics['Historical CVaR / Expected Shortfall (95%)']:.2%}")

    st.markdown("---")

    # Tabs Section
    tab1, tab2, tab3 = st.tabs(["Price History & Drawdowns", "Return Distribution", "Detailed Metrics Table"])

    with tab1:
        fig_price = px.line(df, x=date_col, y=price_col,
                            title="Fund NAV / Price History (Deka-Deutschland Nebenwerte CF)")
        st.plotly_chart(fig_price, use_container_width=True)

        fig_dd = px.area(df, x=date_col, y=drawdowns, title="Under-water Drawdown Chart", labels={'y': 'Drawdown'})
        fig_dd.update_traces(fillcolor="rgba(255, 0, 0, 0.3)", line_color="red")
        st.plotly_chart(fig_dd, use_container_width=True)

    with tab2:
        fig_hist = px.histogram(df.dropna(), x='Daily_Return', nbins=40, title="Daily Return Distribution",
                                marginal="box")
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Indicator", "Value"])
        st.dataframe(metrics_df, use_container_width=True)

except Exception as e:
    st.error(f"Error processing the file: {e}")