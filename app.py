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
  df_raw = pd.read_excel("Trainee-Aufgabe_2026.xlsx", sheet_name="Tabelle1")
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
        "📊 Dashboard & Metrics",
        "📑 Fund Profile & Presentation Data",
        "⚠️ Risk & Opportunities",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Fund:** Deka-Deutschland Nebenwerte CF\n**ISIN:** LU0923076540\n**Fondsmanagement:**"
    " Markus Turnwald"
)

# --- SECTION 1: DASHBOARD & METRICS ---
if section == "📊 Dashboard & Metrics":
  st.title("📊 Deka-Deutschland Nebenwerte CF: Performance 2026")
  st.markdown(
      "Interactive analysis of fund prices, advanced risk metrics, and"
      " historical performance based on official trainee dataset."
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
  sub3.metric("Observation Period", f"{df['Date'].dt.date.nunique()} Days")

  st.markdown("### 📈 Fund Price Evolution Over Time")

  # Interactive Plotly Chart
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
  st.markdown("### 📉 Historical Drawdown Chart")
  fig_dd = px.area(
      df,
      x="Date",
      y=df["Drawdown"] * 100,
      title="Underwater Chart (Drawdown %)",
      labels={"y": "Drawdown (%)", "Date": "Date"},
  )
  fig_dd.update_traces(line_color="red", fillcolor="rgba(255,0,0,0.2)")
  fig_dd.update_layout(
      yaxis_title="Drawdown from Peak (%)", template="plotly_white"
  )
  st.plotly_chart(fig_dd, use_container_width=True)

# --- SECTION 2: FUND PROFILE & PRESENTATION DATA ---
elif section == "📑 Fund Profile & Presentation Data":
  st.title("📑 Fund Profile & Presentation Insights")
  st.markdown(
      "Summary of structural details extracted from the official fund"
      " presentation deck (`ZDF_Deka-Deutschland_Nebenwerte...`)."
  )

  tab1, tab2, tab3 = st.tabs(
      ["Fondsprofil", "Fund Management", "Rating & Structure"]
  )

  with tab1:
    st.subheader("Investment Focus")
    st.write("""
        - **Asset Class:** Equity Fund focusing on German Small & Mid Caps (Nebenwerte).
        - **Benchmark Universe:** MDAX, SDAX, TecDAX, and comparable listed German enterprises.
        - **Investment Objective:** Long-term capital growth through active bottom-up stock selection of high-quality growth and niche market leaders (hidden champions).
        """)

  with tab2:
    st.subheader("Management & Strategy")
    st.write("""
        - **Fund Manager:** Markus Turnwald (Deka Investment GmbH)
        - **Active Management Approach:** High Active Share relative to standard benchmarks to exploit inefficiencies in the German mid-cap segment.
        """)

  with tab3:
    st.subheader("Ratings & Classifications")
    st.write("""
        - **ISIN:** LU0923076540
        - **Share Class:** CF (Retail / Institutional flexible class)
        - Independent agency ratings and monthly reports are maintained in the official Deka reporting framework.
        """)

# --- SECTION 3: RISKS & OPPORTUNITIES ---
elif section == "⚠️ Risk & Opportunities":
  st.title("⚠️ Wesentliche Chancen und Risiken")
  st.markdown(
      "Key opportunities and risk factors associated with investing in German"
      " mid-caps."
  )

  col_c, col_r = st.columns(2)

  with col_c:
    st.success("### 🚀 Chancen (Opportunities)")
    st.markdown("""
        - **High Growth Potential:** Small and mid-caps often outgrow large blue-chip corporations due to higher scalability and nimble business models.
        - **Market Leaders (Hidden Champions):** Access to specialized global market leaders in niche technological and industrial sectors.
        - **Innovation Driver:** Strong exposure to advanced engineering, software, and specialized tech solutions (TecDAX).
        """)

  with col_r:
    st.error("### ⚠️ Risiken (Risks)")
    st.markdown("""
        - **Higher Volatility:** Nebenwerte experience steeper price fluctuations and higher beta compared to broad large-cap indices.
        - **Cyclical Sensitivity:** Export-oriented German mid-caps are vulnerable to global macroeconomic slowdowns and supply chain disruptions.
        - **Liquidity Constraints:** Lower average trading volumes in smaller equities can result in wider bid-ask spreads during market stress.
        """)