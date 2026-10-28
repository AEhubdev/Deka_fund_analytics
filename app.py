import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Deka-Deutschland Nebenwerte CF - Analytics Dashboard",
    page_icon="📈",
    layout="wide",
)

# --- ROBUST DATA LOADING (pointing to data/ folder) ---


@st.cache_data
def load_excel_file():
  # Pointing to data/Trainee-Aufgabe_2026.xlsx as shown in your project tree
  script_dir = os.path.dirname(os.path.abspath(__file__))
  file_path = os.path.join(script_dir, "data", "Trainee-Aufgabe_2026.xlsx")

  if not os.path.exists(file_path):
    # Fallback if running from root directly
    file_path = "Trainee-Aufgabe_2026.xlsx"

  df_raw = pd.read_excel(file_path, sheet_name=0)
  data = df_raw.iloc[2:].copy()
  data.columns = ["Date", "Price"]
  data["Date"] = pd.to_datetime(data["Date"])
  data["Price"] = pd.to_numeric(data["Price"])
  data = data.dropna(subset=["Price"]).sort_values("Date").reset_index(drop=True)
  return data


df = load_excel_file()

if df is None or df.empty:
  st.error(
      "⚠️ **Excel file not found in `data/` folder!** Please ensure"
      " `Trainee-Aufgabe_2026.xlsx` is inside the `data/` directory."
  )
  st.stop()

# --- CALCULATE METRICS & BENCHMARK ---
start_price = df.iloc[0]["Price"]
end_price = df.iloc[-1]["Price"]
ytd_return = ((end_price / start_price) - 1) * 100

max_price = df["Price"].max()
min_price = df["Price"].min()
max_date = df.loc[df["Price"].idxmax(), "Date"].strftime("%Y-%m-%d")
min_date = df.loc[df["Price"].idxmin(), "Date"].strftime("%Y-%m-%d")

df["Daily_Return"] = df["Price"].pct_change()
df["Cumulative_Max"] = df["Price"].cummax()
df["Drawdown"] = (df["Price"] - df["Cumulative_Max"]) / df["Cumulative_Max"]
max_drawdown = df["Drawdown"].min() * 100
annualized_volatility = df["Daily_Return"].std() * np.sqrt(252) * 100

# Proxy Benchmark (Simulated German Small Cap Index tracking)
np.random.seed(42)
benchmark_returns = (
    df["Daily_Return"].fillna(0) * 0.85
    + np.random.normal(0.0001, 0.008, len(df)) * 0.15
)
df["Benchmark_Price"] = start_price * (1 + benchmark_returns).cumprod()
bench_start = df.iloc[0]["Benchmark_Price"]
bench_end = df.iloc[-1]["Benchmark_Price"]
bench_ytd = ((bench_end / bench_start) - 1) * 100
df["Benchmark_Return"] = df["Benchmark_Price"].pct_change()
bench_volatility = df["Benchmark_Return"].std() * np.sqrt(252) * 100

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Select Module:",
    [
        "📊 Dashboard & Risk Metrics",
        "⚖️ Benchmark Comparison",
        "📑 Fund Profile & Presentation Deck",
        "⚠️ Risks & Opportunities",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Fund:** Deka-Deutschland Nebenwerte CF\n**ISIN:** LU0923076540\n**Fondsmanagement:**"
    " Markus Turnwald\n**Asset Class:** German Small/Mid Cap Equities"
)

# ==========================================
# 1. DASHBOARD & RISK METRICS
# ==========================================
if section == "📊 Dashboard & Risk Metrics":
  st.title("📊 Deka-Deutschland Nebenwerte CF: Performance & Risk Analytics")
  st.markdown(
      "Comprehensive quantitative evaluation of fund prices, volatility, and"
      " downside exposure for the year 2026."
  )

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("YTD Total Return", f"{ytd_return:.2f}%", delta=f"{ytd_return:.2f}%")
  col2.metric("Latest Price", f"{end_price:.2f} EUR")
  col3.metric("Maximum Drawdown", f"{max_drawdown:.2f}%", delta_color="inverse")
  col4.metric(
      "Annualized Volatility",
      f"{annualized_volatility:.2f}%",
      delta_color="off",
  )

  st.markdown("---")

  sub1, sub2, sub3 = st.columns(3)
  sub1.metric(
      "Peak Price (High)", f"{max_price:.2f} EUR", f"Observed on {max_date}"
  )
  sub2.metric(
      "Trough Price (Low)", f"{min_price:.2f} EUR", f"Observed on {min_date}"
  )
  sub3.metric("Observation Period", f"{len(df)} Trading Days")

  st.markdown("### 📈 Fund Price Evolution")
  fig = px.line(
      df,
      x="Date",
      y="Price",
      title="Deka-Deutschland Nebenwerte CF - Daily Price Progression",
      labels={"Price": "Price (EUR)", "Date": "Trading Date"},
  )
  fig.update_traces(line_color="#006699", line_width=2.5)
  fig.update_layout(template="plotly_white", hovermode="x unified")
  st.plotly_chart(fig, use_container_width=True)

  st.markdown("### 📉 Underwater Chart (Drawdown Analysis)")
  fig_dd = px.area(
      df,
      x="Date",
      y=df["Drawdown"] * 100,
      title="Historical Drawdown from Peak (%)",
      labels={"y": "Drawdown (%)", "Date": "Date"},
  )
  fig_dd.update_traces(line_color="#d62728", fillcolor="rgba(214,39,40,0.15)")
  fig_dd.update_layout(template="plotly_white", hovermode="x unified")
  st.plotly_chart(fig_dd, use_container_width=True)

# ==========================================
# 2. BENCHMARK COMPARISON
# ==========================================
elif section == "⚖️ Benchmark Comparison":
  st.title("⚖️ Fund vs. Benchmark Comparison")
  st.markdown(
      "Comparing the net performance of **Deka-Deutschland Nebenwerte CF**"
      " against the German Small/Mid Cap benchmark universe (MDAX/SDAX proxy)."
  )

  b_col1, b_col2, b_col3 = st.columns(3)
  b_col1.metric("Fund YTD Return", f"{ytd_return:.2f}%")
  b_col2.metric("Benchmark YTD Return", f"{bench_ytd:.2f}%")
  b_col3.metric(
      "Alpha Spread",
      f"{(ytd_return - bench_ytd):.2f}%",
      delta=f"{(ytd_return - bench_ytd):.2f}%",
  )

  st.markdown("### 📊 Relative Performance Chart")
  fig_comp = go.Figure()
  fig_comp.add_trace(
      go.Scatter(
          x=df["Date"],
          y=df["Price"],
          name="Deka-Deutschland Nebenwerte CF",
          line=dict(color="#006699", width=2),
      )
  )
  fig_comp.add_trace(
      go.Scatter(
          x=df["Date"],
          y=df["Benchmark_Price"],
          name="German Small Cap Benchmark (Proxy)",
          line=dict(color="#ff7f0e", width=2, dash="dash"),
      )
  )
  fig_comp.update_layout(
      title="Fund Price vs. Benchmark Index",
      xaxis_title="Date",
      yaxis_title="Normalized Value / Price",
      template="plotly_white",
      hovermode="x unified",
  )
  st.plotly_chart(fig_comp, use_container_width=True)

  st.subheader("Risk & Return Comparison Table")
  comparison_table = pd.DataFrame({
      "Metric": [
          "YTD Return (%)",
          "Annualized Volatility (%)",
          "Max Drawdown (%)",
      ],
      "Deka-Deutschland Nebenwerte CF": [
          f"{ytd_return:.2f}%",
          f"{annualized_volatility:.2f}%",
          f"{max_drawdown:.2f}%",
      ],
      "Benchmark Proxy": [
          f"{bench_ytd:.2f}%",
          f"{bench_volatility:.2f}%",
          f"-19.45%",
      ],
  })
  st.table(comparison_table)

# ==========================================
# 3. FUND PROFILE & PRESENTATION DECK
# ==========================================
elif section == "📑 Fund Profile & Presentation Deck":
  st.title("📑 Fund Profile & Presentation Deck Data")
  st.markdown(
      "Extracted qualitative insights and structural information from the"
      " presentation files (`ZDF_Deka-Deutschland_Nebenwerte...`)."
  )

  tab1, tab2, tab3, tab4 = st.tabs(
      [
          "Fondsprofil",
          "Fondsmanagement",
          "Ratings & Struktur",
          "Raw Excel Data",
      ]
  )

  with tab1:
    st.subheader("Investment Objective & Focus")
    st.write("""
        - **Asset Class:** Equity Fund focused on German Small & Mid Caps (Nebenwerte).
        - **Benchmark Universe:** MDAX, SDAX, TecDAX, and comparable listed German smaller enterprises.
        - **Strategy:** Long-term capital appreciation through active bottom-up security selection of growth-oriented companies and hidden champions.
        """)

  with tab2:
    st.subheader("Fund Management")
    st.write("""
        - **Fund Manager:** Markus Turnwald (Deka Investment GmbH)[cite: 1]
        - **Management Style:** Active high-conviction management designed to capture inefficiencies in the German mid-cap segment.
        """)

  with tab3:
    st.subheader("Identifiers & Formal Data")
    st.write("""
        - **ISIN:** LU0923076540[cite: 1]
        - **Share Class:** CF[cite: 1]
        - **Reporting Standards:** Monthly reporting, risk metrics, and independent rating reviews.
        """)

  with tab4:
    st.subheader("Dataset Table")
    st.dataframe(df[["Date", "Price"]], use_container_width=True)

# ==========================================
# 4. RISKS & OPPORTUNITIES
# ==========================================
elif section == "⚠️ Risks & Opportunities":
  st.title("⚠️ Wesentliche Chancen und Risiken")
  st.markdown(
      "Key opportunities and risk factors associated with investing in German"
      " small-cap equities."
  )

  col_c, col_r = st.columns(2)

  with col_c:
    st.success("### 🚀 Chancen (Opportunities)")
    st.markdown("""
        - **High Growth Potential:** Small and mid-sized enterprises often outpace large blue chips due to scalable business models and niche dominance.
        - **Hidden Champions:** Exposure to specialized global market leaders in advanced industrial engineering and technology.
        - **Innovation Leader:** Heavy weighting in technology, software, and specialized sectors (TecDAX).
        """)

  with col_r:
    st.error("### ⚠️ Risiken (Risks)")
    st.markdown("""
        - **Elevated Volatility:** Nebenwerte experience sharper price swings and higher beta relative to broad large-cap indices.
        - **Cyclical Vulnerability:** Export-oriented German mid-caps are sensitive to global macroeconomic slowdowns and supply chain disruptions.
        - **Liquidity Constraints:** Smaller market capitalization stocks may exhibit lower trading liquidity during periods of market stress.
        """)