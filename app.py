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
    """Build daily-rebalanced 80/20 wealth, then sample at the Excel dates."""
    daily = indices.loc[(indices.index >= fund_dates[0]) & (indices.index <= fund_dates[-1])].copy()
    missing = daily[daily.isna().any(axis=1)].index
    if len(missing):
        examples = ", ".join(date.strftime("%Y-%m-%d") for date in missing[:5])
        raise RuntimeError(
            f"Yahoo has incomplete MDAX or SDAX closes on {len(missing)} market date(s), including {examples}. "
            "The benchmark is not forward-filled."
        )

    missing_excel_dates = fund_dates.difference(daily.index)
    if len(missing_excel_dates):
        examples = ", ".join(date.strftime("%Y-%m-%d") for date in missing_excel_dates[:5])
        raise RuntimeError(f"Yahoo has no index close on {len(missing_excel_dates)} Excel date(s), including {examples}.")
    if len(daily) < 2 or daily.index[0] != fund_dates[0]:
        raise RuntimeError("Yahoo data does not include the first Excel date as the benchmark baseline.")

    index_returns = daily.pct_change(fill_method=None)
    composite_returns = MDAX_WEIGHT * index_returns["MDAX"] + SDAX_WEIGHT * index_returns["SDAX"]
    composite_returns.iloc[0] = 0.0
    daily["Benchmark_Return"] = composite_returns
    daily["Benchmark_Wealth"] = 100.0 * (1.0 + composite_returns).cumprod()
    sampled = daily.reindex(fund_dates).copy()
    # Recompute sampled-interval returns so they line up with fund observations
    # when an Excel observation is separated by a market holiday or missing day.
    sampled["Benchmark_Return"] = sampled["Benchmark_Wealth"].pct_change(fill_method=None)
    return sampled


def calculate_metrics(returns: pd.Series, wealth: pd.Series) -> dict[str, object]:
    clean = returns.dropna()
    drawdown = wealth / wealth.cummax() - 1.0
    downside = np.minimum(clean.to_numpy(dtype=float), 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(252) * 100.0) if len(clean) else float("nan")
    downside_daily = float(np.sqrt(np.mean(np.square(downside)))) if len(clean) else float("nan")
    sortino = float(clean.mean() / downside_daily * np.sqrt(252)) if np.isfinite(downside_daily) and downside_daily > 0 else float("nan")
    trough_date = drawdown.idxmin()
    peak_date = wealth.loc[:trough_date].idxmax()
    peak_level = float(wealth.loc[peak_date])
    recovered = wealth.loc[trough_date:][wealth.loc[trough_date:] >= peak_level]
    recovery_date = recovered.index[0] if len(recovered) else None
    tail = clean[clean <= clean.quantile(0.05)] if len(clean) >= 20 else pd.Series(dtype=float)
    return {
        "return_pct": float((wealth.iloc[-1] / wealth.iloc[0] - 1.0) * 100.0),
        "volatility_pct": float(clean.std(ddof=1) * np.sqrt(252) * 100.0) if len(clean) > 1 else float("nan"),
        "downside_deviation_pct": downside_deviation,
        "sortino_ratio": sortino,
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "current_drawdown_pct": float(drawdown.iloc[-1] * 100.0),
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "recovery_days": int((recovery_date - trough_date).days) if recovery_date is not None else None,
        "underwater_days": int((wealth.index[-1] - peak_date).days) if recovery_date is None else 0,
        "var_95_pct": float(clean.quantile(0.05) * 100.0) if len(clean) >= 20 else float("nan"),
        "cvar_95_pct": float(tail.mean() * 100.0) if len(tail) else float("nan"),
        "best_day_pct": float(clean.max() * 100.0) if len(clean) else float("nan"),
        "worst_day_pct": float(clean.min() * 100.0) if len(clean) else float("nan"),
        "positive_days_pct": float((clean > 0).mean() * 100.0) if len(clean) else float("nan"),
        "return_intervals": int(len(clean)),
    }


def ytd_window(prices: pd.DataFrame, year: int = 2026) -> tuple[pd.DataFrame, bool]:
    """Return the year plus its last prior observation as a return baseline."""
    in_year = prices.index[prices["Date"].dt.year.eq(year)]
    if len(in_year) == 0:
        raise DataError(f"The Excel workbook has no fund prices for {year}.")
    first_year_idx = int(in_year[0])
    last_year_idx = int(in_year[-1])
    has_prior_base = first_year_idx > 0
    start_idx = first_year_idx - 1 if has_prior_base else first_year_idx
    return prices.loc[start_idx:last_year_idx].copy().reset_index(drop=True), has_prior_base


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:+.2f}%".replace(".", ",")


st.title(f"📈 {FUND_NAME}")
st.caption(f"ISIN {ISIN} · Fund prices from the supplied Excel file · Reference index: 80% MDAX / 20% SDAX")

try:
    source_fund = load_fund_prices()
    fund, has_ytd_baseline = ytd_window(source_fund, 2026)
except DataError as exc:
    st.error(str(exc))
    st.stop()

fund_dates = pd.DatetimeIndex(fund["Date"])
start_date = fund_dates[0]
end_date = fund_dates[-1]
fund["Fund_Return"] = fund["Price"].pct_change(fill_method=None)
fund["Fund_Wealth"] = 100.0 * (fund["Price"] / fund["Price"].iloc[0])
fund_dates_indexed = fund.set_index("Date")
fund_stats = calculate_metrics(fund_dates_indexed["Fund_Return"], fund_dates_indexed["Fund_Wealth"])

st.info(
    f"2026 assessment through latest Excel observation: **{end_date:%d %b %Y}** · "
    f"{len(fund)} observations including the year-start baseline · {fund_stats['return_intervals']} return intervals. "
    + ("The YTD return starts from the last available 2025 price." if has_ytd_baseline else "No pre-2026 price is present, so the exact year-to-date return from the prior year-end cannot be calculated.")
    + " The Excel file contains fund prices only, so fund performance below is a price return and excludes cash distributions."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("2026 YTD price return", pct(fund_stats["return_pct"]), help="Measured from the last available 2025 fund price to the latest Excel price in 2026.")
c2.metric("Latest fund price", f"{fund['Price'].iloc[-1]:.2f} EUR")
c3.metric("Maximum drawdown in 2026", pct(fund_stats["max_drawdown_pct"]))
c4.metric("Annualized volatility", f"{fund_stats['volatility_pct']:.2f}%" if np.isfinite(fund_stats["volatility_pct"]) else "n/a")

direction = "positive" if fund_stats["return_pct"] > 0 else "negative" if fund_stats["return_pct"] < 0 else "flat"
sortino_summary = f"{fund_stats['sortino_ratio']:.2f}" if np.isfinite(fund_stats["sortino_ratio"]) else "n/a"
st.write(
    f"Assessment through {end_date:%d %B %Y}: the fund's 2026 price return is {pct(fund_stats['return_pct'])} ({direction}), "
    f"with annualized volatility of {fund_stats['volatility_pct']:.1f}% and a maximum drawdown of {pct(fund_stats['max_drawdown_pct'])} "
    f"from {fund_stats['peak_date']:%d %b} to {fund_stats['trough_date']:%d %b}. "
    f"Its Sortino ratio is {sortino_summary} using a 0% target; the return and risk path should be considered together. "
    "The benchmark tab adds the same-date 80/20 MDAX/SDAX comparison, subject to the return-basis caveat."
)

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

    r1, r2, r3 = st.columns(3)
    r1.metric("Current drawdown from peak", pct(fund_stats["current_drawdown_pct"]))
    r2.metric("Peak before maximum drawdown", fund_stats["peak_date"].strftime("%d %b %Y"))
    r3.metric("Maximum drawdown trough", fund_stats["trough_date"].strftime("%d %b %Y"))
    if fund_stats["recovery_date"] is not None:
        st.write(f"The maximum drawdown recovered its previous peak on **{fund_stats['recovery_date']:%d %b %Y}**, {fund_stats['recovery_days']} calendar days after the trough.")
    else:
        st.write(f"The maximum drawdown had not regained its prior peak by **{end_date:%d %b %Y}**; the fund remained below that peak for {fund_stats['underwater_days']} calendar days from the peak date.")

    risk1, risk2, risk3, risk4, risk5 = st.columns(5)
    risk1.metric("Annualized downside deviation", f"{fund_stats['downside_deviation_pct']:.2f}%" if np.isfinite(fund_stats["downside_deviation_pct"]) else "n/a")
    risk2.metric("Sortino ratio · 0% target", f"{fund_stats['sortino_ratio']:.2f}" if np.isfinite(fund_stats["sortino_ratio"]) else "n/a")
    risk3.metric("Historical daily VaR · 95%", pct(fund_stats["var_95_pct"]))
    risk4.metric("Historical daily CVaR · 95%", pct(fund_stats["cvar_95_pct"]))
    risk5.metric("Positive observations", f"{fund_stats['positive_days_pct']:.1f}%" if np.isfinite(fund_stats["positive_days_pct"]) else "n/a")
    st.caption("Downside deviation and Sortino use a 0% daily target and are annualized with 252 trading days. Historical VaR is the 5th percentile of observed daily returns; CVaR is the average return in that worst 5% tail. These are backward-looking estimates, not loss limits or forecasts.")

    best_date = fund_dates_indexed["Fund_Return"].idxmax()
    worst_date = fund_dates_indexed["Fund_Return"].idxmin()
    best, worst = st.columns(2)
    best.metric("Best observed daily return", pct(fund_stats["best_day_pct"]), help=f"Observation date: {best_date:%d %b %Y}")
    worst.metric("Worst observed daily return", pct(fund_stats["worst_day_pct"]), help=f"Observation date: {worst_date:%d %b %Y}")

    monthly_source = fund.dropna(subset=["Fund_Return"]).copy()
    monthly = monthly_source.groupby(monthly_source["Date"].dt.to_period("M"))["Fund_Return"].apply(lambda values: (1.0 + values).prod() - 1.0)
    monthly.index = monthly.index.to_timestamp(how="end").normalize()
    monthly_chart = go.Figure()
    monthly_chart.add_trace(go.Bar(x=monthly.index, y=monthly.values * 100.0, marker_color=["#18866B" if value >= 0 else "#C25554" for value in monthly.values], name="Monthly price return"))
    monthly_chart.update_layout(title="Monthly compounded fund price returns", xaxis_title="Month", yaxis_title="Return (%)", template="plotly_white")
    st.plotly_chart(monthly_chart, use_container_width=True)
    monthly_positive = int((monthly > 0).sum())
    st.caption(f"Positive months: {monthly_positive} of {len(monthly)}. The first monthly figure starts at the last available 2025 price, and the last month ends at the latest Excel observation.")

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

        pct_metrics = ["Return", "Annualized volatility", "Annualized downside deviation", "Maximum drawdown", "Current drawdown"]
        fund_risk_values = [fund_stats["return_pct"], fund_stats["volatility_pct"], fund_stats["downside_deviation_pct"], fund_stats["sortino_ratio"], fund_stats["max_drawdown_pct"], fund_stats["current_drawdown_pct"]]
        benchmark_risk_values = [benchmark_stats["return_pct"], benchmark_stats["volatility_pct"], benchmark_stats["downside_deviation_pct"], benchmark_stats["sortino_ratio"], benchmark_stats["max_drawdown_pct"], benchmark_stats["current_drawdown_pct"]]
        risk_labels = ["Return", "Annualized volatility", "Annualized downside deviation", "Sortino ratio (0% target)", "Maximum drawdown", "Current drawdown"]
        format_risk = lambda values: ["n/a" if not np.isfinite(value) else (f"{value:+.2f}%" if label in pct_metrics else f"{value:.2f}") for label, value in zip(risk_labels, values)]
        risk_table = pd.DataFrame({"Metric": risk_labels, "Fund (Excel price series)": format_risk(fund_risk_values), "80/20 Yahoo composite": format_risk(benchmark_risk_values)})
        st.dataframe(risk_table, hide_index=True, use_container_width=True)

        paired_returns = pd.concat(
            [fund_dates_indexed["Fund_Return"].rename("Fund"), benchmark["Benchmark_Return"].rename("Benchmark")], axis=1
        ).dropna()
        if len(paired_returns) > 2 and paired_returns["Benchmark"].var(ddof=1) > 0:
            active_returns = paired_returns["Fund"] - paired_returns["Benchmark"]
            beta = float(paired_returns["Fund"].cov(paired_returns["Benchmark"]) / paired_returns["Benchmark"].var(ddof=1))
            correlation = float(paired_returns["Fund"].corr(paired_returns["Benchmark"]))
            tracking_error = float(active_returns.std(ddof=1) * np.sqrt(252) * 100.0)
            information_ratio = float(active_returns.mean() * 252 / (tracking_error / 100.0)) if tracking_error > 0 else float("nan")
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Return correlation", f"{correlation:.2f}")
            a2.metric("Beta to 80/20 benchmark", f"{beta:.2f}")
            a3.metric("Annualized tracking error", f"{tracking_error:.2f}%")
            a4.metric("Information ratio", f"{information_ratio:.2f}" if np.isfinite(information_ratio) else "n/a")
            st.caption("Beta, correlation, tracking error and information ratio use paired returns from identical Excel dates. Since fund price return and Yahoo performance-index return have different dividend treatment, active-risk measures are indicative rather than like-for-like skill measures.")
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
        "- 2026 YTD price return = latest 2026 Excel price / last 2025 Excel price − 1 (the last 2025 observation is the baseline).\n"
        "- Daily fund return = today's Excel price / previous Excel price − 1.\n"
        "- Benchmark daily return = 0.80 × MDAX daily return + 0.20 × SDAX daily return; benchmark wealth is compounded daily and sampled on the Excel dates.\n"
        "- Both wealth series begin at 100 on the first Excel date and end on the last Excel date.\n"
        "- Annualized volatility = sample standard deviation of daily returns × √252.\n"
        "- Maximum drawdown = lowest daily wealth value relative to its prior running maximum.\n"
        "- Downside deviation and Sortino use a 0% daily target; historical VaR/CVaR use the observed lower 5% tail.\n"
        "- The displayed return difference is measured in percentage points; it is not risk-adjusted alpha."
    )
    export = fund[["Date", "Price", "Fund_Return", "Fund_Wealth"]].copy()
    st.download_button("Download Excel-date fund calculation table", export.to_csv(index=False).encode("utf-8-sig"), file_name="deka_fund_calculations.csv", mime="text/csv")
    st.dataframe(export.tail(20), hide_index=True, use_container_width=True)

st.divider()
st.caption("Fund data is read only from the supplied Excel workbook. Benchmark data is downloaded from Yahoo Finance and cached for one hour. Yahoo data availability and history may change.")
