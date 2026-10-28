import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Deka-Deutschland Nebenwerte CF - Dashboard",
    page_icon="📈",
    layout="wide",
)

# --- LOAD AND CACHE DATA ---


@st.cache_data
def load_data():
  # Get absolute path relative to script directory to avoid FileNotFoundError on Streamlit Cloud
  script_dir = os.path.dirname(os.path.abspath(__file__))
  file_path = os.path.join(script_dir, "Trainee-Aufgabe_2026.xlsx")

  df_raw = pd.read_excel(file_path, sheet_name="Tabelle1")
  # Extract data starting from row index 2
  data = df_raw.iloc[2:].copy()
  data.columns = ["Date", "Price"]
  data["Date"] = pd.to_datetime(data["Date"])
  data["Price"] = pd.to_numeric(data["Price"])
  data = data.sort_values("Date").reset_index(drop=True)
  return data


df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation & Filters")
section = st.sidebar.radio(
    "Go to:",
    [
        "📊 Dashboard & Performance Metrics",
        "📑 Fund Profile & Presentation Data",
        "⚠️ Risks & Opportunities",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Fund:** Deka-Deutschland Nebenwerte CF\n**ISIN:** LU0923076540\n**Fondsmanagement:**"
    " Markus Turnwald"
)

# --- SECTION 1: DASHBOARD & METRICS ---
if section == "📊 Dashboard & Performance Metrics":
  st.title("📊 Deka-Deutschland Nebenwerte CF: Performance 2026")
  st.markdown(
      "Comprehensive interactive analytics for daily fund prices, advanced"
      " risk indicators, and historical drawdowns."
  )

  # Calculate Metrics
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

  # Display Metrics in Columns
  col1, col2, col3, col4 = st.columns(4)
  col1.metric("YTD Return", f"{ytd_return:.2f}%", delta=f"{ytd_return:.2f}%")
  col2.metric("Latest Price", f"{end_price:.2f} EUR")
  col3.metric("Max Drawdown", f"{max_drawdown:.2f}%", delta_color="inverse")
  col4.metric(
      "Ann. Volatility",
      f"{annualized_volatility:.2f}%",
      delta_color="off",
  )

  st.markdown("---")

  # Additional Stats
  sub1, sub2, sub3 = st.columns(3)
  sub1.metric(
      "Highest Price (Peak)", f"{max_price:.2f} EUR", f"Reached on {max_date}"
  )
  sub2.metric(
      "Lowest Price (Trough)", f"{min_price:.2f} EUR", f"Reached on {min_date}"
  )
  sub3.metric("Observation Days", f"{df['Date'].dt.date.nunique()} Trading Days")

  st.markdown("### 📈 Fund Price Evolution Over Time")

  # Interactive Plotly Chart for Prices
  fig = px.line(
      df,
      x="Date",
      y="Price",
      title="Deka-Deutschland Nebenwerte CF - Daily Fondspreise (2026)",
      labels={"Price": "Price (EUR)", "Date": "Trading Date"},
  )
  fig.update_traces(line_color="#1f77b4", line_width=2)
  fig.update_layout(
      xaxis_title="Date",
      yaxis_title="Price in EUR",
      hovermode="x unified",
      template="plotly_white",
  )
  st.plotly_chart(fig, use_container_width=True)

  # Drawdown Chart
  st.markdown("### 📉 Historical Drawdown (Underwater Chart)")
  fig_dd = px.area(
      df,
      x="Date",
      y=df["Drawdown"] * 100,
      title="Drawdown Percentage from Peak",
      labels={"y": "Drawdown (%)", "Date": "Date"},
  )
  fig_dd.update_traces(line_color="#d62728", fillcolor="rgba(214,39,40,0.2)")
  fig_dd.update_layout(
      yaxis_title="Drawdown (%)", template="plotly_white", hovermode="x unified"
  )
  st.plotly_chart(fig_dd, use_container_width=True)

# --- SECTION 2: FUND PROFILE & PRESENTATION DATA ---
elif section == "📑 Fund Profile & Presentation Data":
  st.title("📑 Fund Profile & Presentation Insights")
  st.markdown(
      "Detailed structural breakdown sourced from the official presentation"
      " deck (`ZDF_Deka-Deutschland_Nebenwerte...`)."
  )

  tab1, tab2, tab3, tab4 = st.tabs(
      [
          "Fondsprofil",
          "Fondsmanagement",
          "Ratings & Struktur",
          "Raw Excel Data Preview",
      ]
  )

  with tab1:
    st.subheader("Investment Focus & Strategy")
    st.write("""
        - **Asset Class:** Equity Fund focusing on German Small & Mid Caps (Nebenwerte).
        - **Benchmark Universe:** MDAX, SDAX, TecDAX, and comparable listed smaller enterprises.
        - **Investment Objective:** Long-term capital growth through active bottom-up stock selection of hidden champions and high-growth innovators.
        """)

  with tab2:
    st.subheader("Management Details")
    st.write("""
        - **Fund Manager:** Markus Turnwald (Deka Investment GmbH)
        - **Active Management Approach:** High Active Share relative to standard benchmarks to capitalize on market inefficiencies within the German mid-cap segment.
        """)

  with tab3:
    st.subheader("Key Identifiers & Classifications")
    st.write("""
        - **ISIN:** LU0923076540
        - **Share Class:** CF
        - **Reporting Framework:** Regular monthly reporting and independent rating agency evaluations.
        """)

  with tab4:
    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

# --- SECTION 3: RISKS & OPPORTUNITIES ---
elif section == "⚠️ Risks & Opportunities":
  st.title("⚠️ Wesentliche Chancen und Risiken")
  st.markdown(
      "Core opportunities and inherent risk factors associated with German"
      " small and mid-cap equities."
  )

  col_c, col_r = st.columns(2)

  with col_c:
    st.success("### 🚀 Chancen (Opportunities)")
    st.markdown("""
        - **High Growth Potential:** Smaller companies often outgrow large-cap blue chips due to agile structures and niche market expansion.
        - **Hidden Champions:** Direct exposure to global market leaders in specialized technical and industrial sectors.
        - **Innovation Driver:** Strong weight in technology, software, and advanced engineering (TecDAX).
        """)

  with col_r:
    st.error("### ⚠️ Risiken (Risks)")
    st.markdown("""
        - **Higher Volatility:** Nebenwerte exhibit higher price fluctuations and beta relative to large-cap indices (DAX).
        - **Macro & Cyclical Sensitivity:** Export-oriented mid-caps are sensitive to global economic slowdowns and supply chain constraints.
        - **Liquidity Risk:** Smaller capitalization stocks can experience reduced trading liquidity during periods of market stress.
        """)