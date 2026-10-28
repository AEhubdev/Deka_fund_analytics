"""Deka-Deutschland Nebenwerte CF performance dashboard.

Fund prices: data/Trainee-Aufgabe_2026.xlsx
Reference index: 80% MDAX + 20% SDAX, using Yahoo Finance ^MDAXI and ^SDAXI.
The composite return is calculated from the two daily index returns and rebalanced
back to 80/20 each trading day. The Excel file has prices only, so the fund result
is a price return and excludes cash distributions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

FUND_NAME = "Deka-Deutschland Nebenwerte CF"
ISIN = "LU0923076540"
MDAX_WEIGHT = 0.80
SDAX_WEIGHT = 0.20
MDAX_TICKER = "^MDAXI"
SDAX_TICKER = "^SDAXI"
FUND_FILE = Path(__file__).resolve().parent / "data" / "Trainee-Aufgabe_2026.xlsx"

# Static fund facts transcribed from the supplied Deka presentation.
# Portfolio composition and ratings are a dated snapshot, not live holdings.
PROFILE_AS_OF = "30.09.2026"
INDUSTRIES = {
    "Industrie": 20.6,
    "Technologie": 20.0,
    "Chemie": 9.0,
    "Öl & Gas": 6.7,
    "Automobile": 6.2,
    "Rohstoffe": 5.0,
    "Medien": 4.6,
    "Konsumgüter": 4.4,
    "Tourismus & Freizeit": 3.6,
    "Einzelhandel": 3.4,
}
TOP_HOLDINGS = {
    "Knorr-Bremse AG": 4.1,
    "Nordex SE": 3.8,
    "Aurubis AG": 3.4,
    "Evonik Industries AG": 3.2,
    "AUTO1 Group SE": 3.1,
    "HENSOLDT AG": 3.1,
    "Nemetschek SE": 2.9,
    "thyssenkrupp AG": 2.9,
    "Siltronic AG": 2.7,
    "Bechtle AG": 2.5,
}

st.set_page_config(
    page_title=f"{FUND_NAME} | Wertentwicklung",
    page_icon="📈",
    layout="wide",
)


class DataError(ValueError):
    """Readable error for unusable source data."""


def load_fund_prices(path: Path = FUND_FILE) -> pd.DataFrame:
    """Read the supplied workbook's Date/Price series, which has a prompt above it."""
    if not path.exists():
        raise DataError(f"Excel file not found: {path}")
    try:
        raw = pd.read_excel(path, sheet_name=0, header=None)
    except Exception as exc:
        raise DataError(f"Could not read the Excel file: {exc}") from exc

    header_row = None
    for row_number, row in raw.iterrows():
        first_two = [str(value).strip().lower() for value in row.iloc[:2].tolist()]
        if any(value in {"datum", "date"} for value in first_two):
            header_row = row_number
            break
    if header_row is None:
        raise DataError("Could not locate the Date/Datum header in the Excel workbook.")

    prices = raw.iloc[header_row + 1 :, :2].copy()
    prices.columns = ["Date", "Price"]
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce", dayfirst=True)
    prices["Price"] = pd.to_numeric(prices["Price"], errors="coerce")
    prices = prices.dropna(subset=["Date", "Price"])
    prices["Date"] = prices["Date"].dt.normalize()
    prices = prices.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

    if len(prices) < 2:
        raise DataError("The Excel file must contain at least two valid fund prices.")
    if (prices["Price"] <= 0).any():
        raise DataError("Fund prices must be positive.")
    return prices


@st.cache_data(ttl=60 * 60, show_spinner=False)
def download_yahoo_indices(start_date: str, end_date_exclusive: str) -> pd.DataFrame:
    """Download daily Yahoo closes for the Excel period (end date is exclusive)."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is missing. Add it to requirements.txt and install the dependencies.") from exc

    try:
        data = yf.download(
            tickers=[MDAX_TICKER, SDAX_TICKER],
            start=start_date,
            end=end_date_exclusive,
            interval="1d",
            auto_adjust=False,
            actions=False,
            group_by="ticker",
            progress=False,
            threads=False,
            timeout=25,
        )
    except Exception as exc:
        raise RuntimeError(f"Yahoo Finance download failed: {exc}") from exc
    if data is None or data.empty:
        raise RuntimeError("Yahoo Finance returned no index data for the Excel date range.")

    def close_series(ticker: str, output_name: str) -> pd.Series:
        matches = []
        for column in data.columns:
            parts = column if isinstance(column, tuple) else (column,)
            normalised = {str(part).strip().casefold() for part in parts}
            if ticker.casefold() in normalised and "close" in normalised:
                matches.append(column)
        if len(matches) != 1:
            raise RuntimeError(f"Could not find one daily Close series for {ticker} in Yahoo's response.")
        series = pd.to_numeric(data[matches[0]], errors="coerce")
        series.index = pd.to_datetime(series.index, utc=True).tz_convert(None).normalize()
        series = series[~series.index.duplicated(keep="last")].sort_index()
        series.name = output_name
        return series

    indices = pd.concat(
        [close_series(MDAX_TICKER, "MDAX"), close_series(SDAX_TICKER, "SDAX")], axis=1
    ).sort_index()
    if indices.empty or (indices <= 0).any().any():
        raise RuntimeError("Yahoo Finance index closes are empty or invalid.")
    return indices


def make_benchmark(indices: pd.DataFrame, fund_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a daily rebalanced 80/20 composite on exactly the Excel dates."""
    aligned = indices.reindex(fund_dates)
    missing = aligned[aligned.isna().any(axis=1)].index
    if len(missing):
        examples = ", ".join(date.strftime("%Y-%m-%d") for date in missing[:5])
        raise RuntimeError(
            f"Yahoo has no MDAX or SDAX close on {len(missing)} Excel date(s), including {examples}. "
            "No dates were silently forward-filled."
        )

    index_returns = aligned.pct_change(fill_method=None)
    composite_returns = MDAX_WEIGHT * index_returns["MDAX"] + SDAX_WEIGHT * index_returns["SDAX"]
    # The first Excel date is the common baseline, so no return is applied on it.
    composite_returns.iloc[0] = 0.0
    aligned["Benchmark_Return"] = composite_returns
    aligned["Benchmark_Wealth"] = 100.0 * (1.0 + composite_returns).cumprod()
    return aligned


def calculate_metrics(returns: pd.Series, wealth: pd.Series) -> dict[str, float | int]:
    clean = returns.dropna()
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "return_pct": float((wealth.iloc[-1] / wealth.iloc[0] - 1.0) * 100.0),
        "volatility_pct": float(clean.std(ddof=1) * np.sqrt(252) * 100.0) if len(clean) > 1 else float("nan"),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "positive_days_pct": float((clean > 0).mean() * 100.0) if len(clean) else float("nan"),
        "return_intervals": int(len(clean)),
    }


def pct(value: float) -> str:
    return f"{value:+.2f}%".replace(".", ",")


st.title(f"📈 {FUND_NAME}")
st.caption(f"ISIN {ISIN} · Fund prices from the supplied Excel file · Reference index: 80% MDAX / 20% SDAX")

try:
    fund = load_fund_prices()
except DataError as exc:
    st.error(str(exc))
    st.stop()

fund_dates = pd.DatetimeIndex(fund["Date"])
start_date = fund_dates[0]
end_date = fund_dates[-1]
fund["Fund_Return"] = fund["Price"].pct_change(fill_method=None)
fund["Fund_Wealth"] = 100.0 * (fund["Price"] / fund["Price"].iloc[0])
fund_stats = calculate_metrics(fund["Fund_Return"], fund["Fund_Wealth"])

st.info(
    f"Excel period: **{start_date:%d %b %Y} to {end_date:%d %b %Y}** · "
    f"{len(fund)} fund-price observations · {fund_stats['return_intervals']} return intervals. "
    "The Excel file contains fund prices only, so fund performance below is a price return and excludes cash distributions."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fund price return", pct(fund_stats["return_pct"]))
c2.metric("Latest fund price", f"{fund['Price'].iloc[-1]:.2f} EUR")
c3.metric("Maximum drawdown", pct(fund_stats["max_drawdown_pct"]))
c4.metric("Annualized volatility", f"{fund_stats['volatility_pct']:.2f}%" if np.isfinite(fund_stats["volatility_pct"]) else "n/a")

if fund_stats["return_pct"] > 0 and fund_stats["max_drawdown_pct"] <= -10:
    st.write("The fund ended the Excel period with a positive price return, after a substantial decline from an earlier peak. The end-point return alone understates the path risk.")
elif fund_stats["return_pct"] > 0:
    st.write("The fund ended the Excel period with a positive price return. Volatility and drawdown provide additional context for the route to that result.")
else:
    st.write("The fund ended the Excel period below its starting price. The drawdown shows the largest decline from a prior peak during the observed period.")

profile_tab, fund_tab, benchmark_tab, data_tab = st.tabs(["Fund profile", "Fund performance", "Benchmark comparison", "Data and methodology"])

with profile_tab:
    st.subheader("Fund profile and portfolio snapshot")
    st.caption(f"Transcribed from the supplied Deka presentation · composition and ratings as of {PROFILE_AS_OF}")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Fund launch", "03.06.2013")
    p2.metric("Distribution", "Distributing")
    p3.metric("Minimum investment", "€25")
    p4.metric("Financial year-end", "31 May")

    left, right = st.columns(2)
    with left:
        industry_frame = pd.DataFrame({"Industry": list(INDUSTRIES), "Share (%)": list(INDUSTRIES.values())}).sort_values("Share (%)")
        industry_chart = go.Figure(go.Bar(x=industry_frame["Share (%)"], y=industry_frame["Industry"], orientation="h", marker_color="#117C83", text=industry_frame["Share (%)"].map(lambda x: f"{x:.1f}%"), textposition="outside"))
        industry_chart.update_layout(title=f"Largest industries · {PROFILE_AS_OF}", xaxis_title="Portfolio share (%)", yaxis_title="", template="plotly_white", margin={"l": 10, "r": 40, "t": 55, "b": 35})
        st.plotly_chart(industry_chart, use_container_width=True)
    with right:
        holding_frame = pd.DataFrame({"Security": list(TOP_HOLDINGS), "Share (%)": list(TOP_HOLDINGS.values())}).sort_values("Share (%)")
        holdings_chart = go.Figure(go.Bar(x=holding_frame["Share (%)"], y=holding_frame["Security"], orientation="h", marker_color="#D69E2E", text=holding_frame["Share (%)"].map(lambda x: f"{x:.1f}%"), textposition="outside"))
        holdings_chart.update_layout(title=f"Top ten holdings · {PROFILE_AS_OF}", xaxis_title="Portfolio share (%)", yaxis_title="", template="plotly_white", margin={"l": 10, "r": 40, "t": 55, "b": 35})
        st.plotly_chart(holdings_chart, use_container_width=True)

    top_sector_total = sum(INDUSTRIES.values())
    top_holding_total = sum(TOP_HOLDINGS.values())
    st.write(f"The ten largest industry groups shown represent **{top_sector_total:.1f}%** of the portfolio; the ten largest holdings represent **{top_holding_total:.1f}%**. The industry chart is a partial breakdown of the largest groups, not the full portfolio allocation.")
    st.info("These are static presentation figures dated 30 September 2026. They are included as context for the Excel-period performance assessment and should not be read as current or as holdings on every date in the performance series.")

    with st.expander("Costs and independent ratings shown in the presentation"):
        st.markdown("- Annual management fee: **1.86%**; transaction costs: **0.40%**; performance fee: **0.41%** under specified conditions (presentation cost snapshot dated 01.10.2026).\n- The presentation shows a maximum issue surcharge of **3.610%** in its performance explanation. Entry charges depend on the channel and are excluded from the fund-price return calculated here.\n- Ratings panel dated 30.09.2026: Morningstar **3/5 stars**; Scope **C**. These are the presentation’s displayed ratings, not a fresh rating check.")

with fund_tab:
    performance = go.Figure()
    performance.add_trace(go.Scatter(x=fund["Date"], y=fund["Fund_Wealth"], mode="lines", name="Fund price", line={"color": "#117C83", "width": 3}))
    performance.update_layout(title="Growth of 100 from the first Excel price", xaxis_title="Date", yaxis_title="Value (first Excel price = 100)", template="plotly_white", hovermode="x unified")
    st.plotly_chart(performance, use_container_width=True)

    drawdown = fund["Fund_Wealth"] / fund["Fund_Wealth"].cummax() - 1.0
    dd_chart = go.Figure()
    dd_chart.add_trace(go.Scatter(x=fund["Date"], y=drawdown * 100.0, mode="lines", fill="tozeroy", name="Drawdown", line={"color": "#C25554", "width": 2}))
    dd_chart.update_layout(title="Fund drawdown from prior peak", xaxis_title="Date", yaxis_title="Drawdown (%)", template="plotly_white", hovermode="x unified")
    st.plotly_chart(dd_chart, use_container_width=True)

    monthly_source = fund.dropna(subset=["Fund_Return"]).copy()
    monthly = monthly_source.groupby(monthly_source["Date"].dt.to_period("M"))["Fund_Return"].apply(lambda values: (1.0 + values).prod() - 1.0)
    monthly.index = monthly.index.to_timestamp(how="end").normalize()
    monthly_chart = go.Figure()
    monthly_chart.add_trace(go.Bar(x=monthly.index, y=monthly.values * 100.0, marker_color=["#18866B" if value >= 0 else "#C25554" for value in monthly.values], name="Monthly price return"))
    monthly_chart.update_layout(title="Monthly compounded fund price returns", xaxis_title="Month", yaxis_title="Return (%)", template="plotly_white")
    st.plotly_chart(monthly_chart, use_container_width=True)

with benchmark_tab:
    st.subheader("Deka reference index: 80% MDAX / 20% SDAX")
    st.caption("Yahoo Finance daily closes for ^MDAXI and ^SDAXI. The composite is calculated as 80% × MDAX daily return + 20% × SDAX daily return and rebalanced to those weights each trading day.")
    benchmark_error = None
    benchmark = None
    try:
        end_exclusive = (end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        with st.spinner("Loading Yahoo Finance index data for the Excel dates…"):
            index_data = download_yahoo_indices(start_date.strftime("%Y-%m-%d"), end_exclusive)
        benchmark = make_benchmark(index_data, fund_dates)
    except Exception as exc:
        benchmark_error = str(exc)

    if benchmark_error:
        st.error(
            "A same-date benchmark comparison is unavailable. No synthetic values or substitute dates are used. "
            f"Details: {benchmark_error}"
        )
    else:
        fund_return = fund_stats["return_pct"]
        benchmark_return = float(benchmark["Benchmark_Wealth"].iloc[-1] - 100.0)
        gap_pp = fund_return - benchmark_return
        benchmark_stats = calculate_metrics(benchmark["Benchmark_Return"], benchmark["Benchmark_Wealth"])

        st.warning(
            "Return-basis caveat: the Excel file contains fund prices only, while Yahoo's default MDAX/SDAX series are performance indices. "
            "The benchmark includes dividends; the fund price return shown here does not. The return gap is therefore not a like-for-like total-return comparison and is not alpha."
        )
        st.warning(
            "The app calculates the 80/20 composite by daily rebalancing. Deka's public reference-index description gives the component weights, but the Excel file does not provide an official composite return series or its detailed calculation convention."
        )

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Fund price return", pct(fund_return))
        b2.metric("Yahoo 80/20 benchmark return", pct(benchmark_return))
        b3.metric("Difference (percentage points)", f"{gap_pp:+.2f} pp".replace(".", ","))
        b4.metric("Benchmark volatility", f"{benchmark_stats['volatility_pct']:.2f}%" if np.isfinite(benchmark_stats["volatility_pct"]) else "n/a")

        comparison = go.Figure()
        comparison.add_trace(go.Scatter(x=fund["Date"], y=fund["Fund_Wealth"], mode="lines", name="Fund price return", line={"color": "#117C83", "width": 3}))
        comparison.add_trace(go.Scatter(x=benchmark.index, y=benchmark["Benchmark_Wealth"], mode="lines", name="80% MDAX / 20% SDAX", line={"color": "#D69E2E", "width": 2}))
        comparison.update_layout(title="Fund and benchmark over identical Excel dates", xaxis_title="Date", yaxis_title="Value (first Excel date = 100)", template="plotly_white", hovermode="x unified")
        st.plotly_chart(comparison, use_container_width=True)

        risk_table = pd.DataFrame(
            {
                "Metric": ["Return", "Annualized volatility", "Maximum drawdown"],
                "Fund (Excel price series)": [fund_stats["return_pct"], fund_stats["volatility_pct"], fund_stats["max_drawdown_pct"]],
                "80/20 Yahoo composite": [benchmark_stats["return_pct"], benchmark_stats["volatility_pct"], benchmark_stats["max_drawdown_pct"]],
            }
        )
        st.dataframe(risk_table.style.format({"Fund (Excel price series)": "{:+.2f}%", "80/20 Yahoo composite": "{:+.2f}%"}), hide_index=True, use_container_width=True)
        if gap_pp > 0.05:
            st.write(f"On the same Excel dates, the fund's price return was {gap_pp:.2f} percentage points above the calculated 80/20 benchmark. Interpret this directionally because the return bases differ.")
        elif gap_pp < -0.05:
            st.write(f"On the same Excel dates, the fund's price return was {abs(gap_pp):.2f} percentage points below the calculated 80/20 benchmark. Interpret this directionally because the return bases differ.")
        else:
            st.write("On the same Excel dates, the fund price return and calculated 80/20 benchmark return were close. The return bases still differ.")

with data_tab:
    st.subheader("Source data")
    st.write(f"Fund source: `{FUND_FILE.name}`")
    st.write(f"Fund observations: **{len(fund)}**, from **{start_date:%Y-%m-%d}** through **{end_date:%Y-%m-%d}**.")
    st.write("Benchmark source: Yahoo Finance via `yfinance`, symbols `^MDAXI` and `^SDAXI`.")
    st.write("Deka reference weights: 80% MDAX and 20% SDAX.")
    st.markdown(
        "**Calculations**\n"
        "- Fund price return = last Excel price / first Excel price − 1.\n"
        "- Daily fund return = today's Excel price / previous Excel price − 1.\n"
        "- Benchmark daily return = 0.80 × MDAX daily return + 0.20 × SDAX daily return.\n"
        "- Both wealth series begin at 100 on the first Excel date and end on the last Excel date.\n"
        "- Annualized volatility = sample standard deviation of daily returns × √252.\n"
        "- Maximum drawdown = lowest daily wealth value relative to its prior running maximum.\n"
        "- The displayed return difference is measured in percentage points; it is not risk-adjusted alpha."
    )
    export = fund[["Date", "Price", "Fund_Return", "Fund_Wealth"]].copy()
    st.download_button("Download Excel-date fund calculation table", export.to_csv(index=False).encode("utf-8-sig"), file_name="deka_fund_calculations.csv", mime="text/csv")
    st.dataframe(export.tail(20), hide_index=True, use_container_width=True)

st.divider()
st.caption("Fund data is read only from the supplied Excel workbook. Benchmark data is downloaded from Yahoo Finance and cached for one hour. Yahoo data availability and history may change.")
