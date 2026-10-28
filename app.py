"""Deka-Deutschland Nebenwerte CF performance dashboard.

Benchmark reference weights: 80% MDAX and 20% SDAX, as stated by Deka.
Yahoo Finance symbols: ^MDAXI and ^SDAXI. The composite below is explicitly
calculated as a daily rebalanced 80/20 portfolio of those index returns.
"""

from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


FUND_NAME = "Deka-Deutschland Nebenwerte CF"
ISIN = "LU0923076540"
MDAX_WEIGHT = 0.80
SDAX_WEIGHT = 0.20
DEFAULT_MDAX_TICKER = "^MDAXI"
DEFAULT_SDAX_TICKER = "^SDAXI"
DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_FUND_FILE = DATA_DIR / "Trainee-Aufgabe_2026.xlsx"

st.set_page_config(
    page_title=f"{FUND_NAME} | Performanceanalyse",
    page_icon="📈",
    layout="wide",
)


class DataError(ValueError):
    """Raised when an input file cannot support a reliable calculation."""


def _normalise_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _read_uploaded_or_default(uploaded_file, default_path: Path) -> tuple[bytes, str]:
    if uploaded_file is not None:
        return uploaded_file.getvalue(), uploaded_file.name
    if not default_path.exists():
        raise DataError(
            f"Fund data file not found: {default_path}. Upload an Excel or CSV file."
        )
    return default_path.read_bytes(), default_path.name


def _parse_fund_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read Date/Price from the supplied workbook layout or a normal CSV."""
    suffix = Path(filename).suffix.lower()
    try:
        if suffix in {".xlsx", ".xlsm", ".xls"}:
            raw = pd.read_excel(BytesIO(file_bytes), sheet_name=0, header=None)
            header_row = None
            for idx, row in raw.iterrows():
                labels = {_normalise_column_name(v) for v in row.iloc[:2].tolist()}
                if any(v in labels for v in {"date", "datum"}):
                    header_row = idx
                    break
            if header_row is None:
                # The current training workbook has its date/price columns on row 3.
                header_row = 2
            data = raw.iloc[header_row + 1 :, :2].copy()
            data.columns = ["Date", "Price"]
        elif suffix == ".csv":
            raw = pd.read_csv(BytesIO(file_bytes))
            columns = {_normalise_column_name(c): c for c in raw.columns}
            date_col = next(
                (columns[k] for k in ("date", "datum") if k in columns), None
            )
            price_col = next(
                (columns[k] for k in ("price", "fondspreis", "nav", "close") if k in columns),
                None,
            )
            if date_col is None or price_col is None:
                raise DataError("CSV must have Date/Datum and Price/Fondspreis columns.")
            data = raw[[date_col, price_col]].copy()
            data.columns = ["Date", "Price"]
        else:
            raise DataError("Fund data must be an .xlsx, .xlsm, .xls, or .csv file.")
    except DataError:
        raise
    except Exception as exc:  # pandas raises several engine-specific exceptions
        raise DataError(f"Could not read the fund file: {exc}") from exc

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce", dayfirst=True)
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    data = data.dropna(subset=["Date", "Price"])
    data["Date"] = data["Date"].dt.normalize()
    data = data.sort_values("Date").drop_duplicates("Date", keep="last")
    if data.empty:
        raise DataError("No valid Date/Price observations were found.")
    if (data["Price"] <= 0).any():
        raise DataError("Fund prices must be positive numbers.")
    return data.reset_index(drop=True)


def _parse_distribution_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read ex-date cash distributions per share from an optional CSV."""
    if Path(filename).suffix.lower() != ".csv":
        raise DataError("Distribution history must be a CSV file.")
    try:
        raw = pd.read_csv(BytesIO(file_bytes))
    except Exception as exc:
        raise DataError(f"Could not read the distribution file: {exc}") from exc
    columns = {_normalise_column_name(c): c for c in raw.columns}
    date_col = next((columns[k] for k in ("date", "datum", "exdate") if k in columns), None)
    amount_col = next(
        (columns[k] for k in ("distribution", "distributionpershare", "amount", "ausschttung") if k in columns),
        None,
    )
    if date_col is None or amount_col is None:
        raise DataError("Distribution CSV needs Date (ex-date) and Distribution (EUR/share) columns.")
    dist = raw[[date_col, amount_col]].copy()
    dist.columns = ["Date", "Distribution"]
    dist["Date"] = pd.to_datetime(dist["Date"], errors="coerce", dayfirst=True).dt.normalize()
    dist["Distribution"] = pd.to_numeric(dist["Distribution"], errors="coerce")
    dist = dist.dropna().groupby("Date", as_index=False)["Distribution"].sum()
    if (dist["Distribution"] < 0).any():
        raise DataError("Distribution values must be non-negative.")
    return dist


def _add_fund_returns(
    fund: pd.DataFrame, distributions: pd.DataFrame | None
) -> tuple[pd.DataFrame, str]:
    out = fund.copy()
    out["Distribution"] = 0.0
    if distributions is not None and not distributions.empty:
        in_range = distributions.loc[
            distributions["Date"].between(out["Date"].min(), out["Date"].max())
        ].copy()
        unknown_dates = sorted(set(in_range["Date"]) - set(out["Date"]))
        if unknown_dates:
            dates = ", ".join(d.strftime("%Y-%m-%d") for d in unknown_dates[:5])
            raise DataError(
                "Every distribution ex-date must match a fund-price date. "
                f"Unmatched date(s): {dates}"
            )
        dist_map = in_range.set_index("Date")["Distribution"]
        out["Distribution"] = out["Date"].map(dist_map).fillna(0.0)
        out["Fund_Return"] = (out["Price"] + out["Distribution"]) / out["Price"].shift(1) - 1
        basis = "Total return, including supplied cash distributions"
    else:
        out["Fund_Return"] = out["Price"].pct_change()
        basis = "Price return only; distribution history was not supplied"
    return out, basis


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _download_yahoo_indices(
    start_iso: str, end_iso: str, mdax_ticker: str, sdax_ticker: str
) -> pd.DataFrame:
    """Download Yahoo Finance index closes, including a lookback for the baseline."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Add 'yfinance' to requirements.txt and install dependencies."
        ) from exc

    start = pd.Timestamp(start_iso) - pd.Timedelta(days=14)
    end_exclusive = pd.Timestamp(end_iso) + pd.Timedelta(days=1)
    tickers = [mdax_ticker.strip(), sdax_ticker.strip()]
    try:
        downloaded = yf.download(
            tickers=tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end_exclusive.strftime("%Y-%m-%d"),
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
    if downloaded is None or downloaded.empty:
        raise RuntimeError("Yahoo Finance returned no index observations for this period.")

    def extract_close(ticker: str) -> pd.Series:
        matches = []
        for column in downloaded.columns:
            parts = column if isinstance(column, tuple) else (column,)
            normal = {_normalise_column_name(part) for part in parts}
            if _normalise_column_name(ticker) in normal and "close" in normal:
                matches.append(column)
        # A single-ticker/older-yfinance response may omit ticker in its columns.
        if not matches and len(tickers) == 1 and "Close" in downloaded.columns:
            matches = ["Close"]
        if len(matches) != 1:
            raise RuntimeError(
                f"Could not identify one Close series for Yahoo ticker {ticker}. "
                "Check the ticker symbol and Yahoo Finance response."
            )
        series = pd.to_numeric(downloaded[matches[0]], errors="coerce")
        index = pd.to_datetime(series.index, utc=True).tz_convert(None).normalize()
        series.index = index
        series = series[~series.index.duplicated(keep="last")].sort_index()
        series.name = ticker
        return series

    mdax = extract_close(tickers[0]).rename("MDAX")
    sdax = extract_close(tickers[1]).rename("SDAX")
    indices = pd.concat([mdax, sdax], axis=1).dropna(how="any").sort_index()
    if indices.empty or (indices <= 0).any().any():
        raise RuntimeError("Yahoo data is missing or contains invalid index levels.")
    return indices


def _make_benchmark(
    index_levels: pd.DataFrame, baseline_date: pd.Timestamp, fund_dates: pd.Series
) -> pd.DataFrame:
    """Build a daily rebalanced 80/20 index composite, normalized at baseline."""
    levels = index_levels[["MDAX", "SDAX"]].dropna().sort_index()
    returns = levels.pct_change()
    composite_returns = MDAX_WEIGHT * returns["MDAX"] + SDAX_WEIGHT * returns["SDAX"]
    composite = (1 + composite_returns.fillna(0)).cumprod() * 100.0

    prior = composite.loc[composite.index <= baseline_date]
    if prior.empty:
        raise RuntimeError(
            f"Yahoo data does not include an index close on or before {baseline_date:%Y-%m-%d}."
        )
    base_value = float(prior.iloc[-1])
    composite = composite / base_value * 100.0
    aligned = composite.reindex(pd.DatetimeIndex(fund_dates), method="ffill")
    aligned.index = pd.DatetimeIndex(fund_dates)
    if aligned.isna().any():
        first_missing = aligned[aligned.isna()].index[0]
        raise RuntimeError(f"No benchmark close available by {first_missing:%Y-%m-%d}.")
    result = pd.DataFrame({"Benchmark_Wealth": aligned})
    result["Benchmark_Return"] = result["Benchmark_Wealth"].pct_change()
    result.iloc[0, result.columns.get_loc("Benchmark_Return")] = np.nan
    return result


def _format_pct(value: float, decimals: int = 2) -> str:
    return f"{value:+.{decimals}f}%".replace(".", ",")


def _metrics(returns: pd.Series, wealth: pd.Series) -> dict[str, float | int]:
    clean_returns = returns.dropna()
    drawdown = wealth / wealth.cummax() - 1
    volatility = (
        float(clean_returns.std(ddof=1) * np.sqrt(252) * 100)
        if len(clean_returns) > 1
        else float("nan")
    )
    return {
        "return_pct": float((wealth.iloc[-1] / wealth.iloc[0] - 1) * 100),
        "volatility_pct": volatility,
        "max_drawdown_pct": float(drawdown.min() * 100),
        "positive_days_pct": float((clean_returns > 0).mean() * 100) if len(clean_returns) else float("nan"),
        "observations": int(len(wealth)),
        "return_intervals": int(len(clean_returns)),
    }


st.title(f"📈 {FUND_NAME}")
st.caption(f"ISIN {ISIN} · Fund return from supplied daily prices · Reference index: 80% MDAX / 20% SDAX")

with st.sidebar:
    st.header("Data and period")
    fund_upload = st.file_uploader("Fund prices (.xlsx or .csv)", type=["xlsx", "xlsm", "xls", "csv"])
    dist_upload = st.file_uploader(
        "Optional fund distributions (.csv)",
        type=["csv"],
        help="Use Date (ex-date) and Distribution (EUR per share). Without this file, fund results are price returns only.",
    )
    dist_complete = st.checkbox(
        "I confirm the distribution file includes every ex-date in the selected period",
        value=False,
        disabled=dist_upload is None,
    )

try:
    fund_bytes, fund_filename = _read_uploaded_or_default(fund_upload, DEFAULT_FUND_FILE)
    fund_prices = _parse_fund_bytes(fund_bytes, fund_filename)
    distributions = (
        _parse_distribution_bytes(dist_upload.getvalue(), dist_upload.name)
        if dist_upload is not None
        else None
    )
    fund, fund_basis = _add_fund_returns(fund_prices, distributions)
    if distributions is not None and not distributions.empty:
        if dist_complete:
            fund_basis = "Total return, with user-confirmed distribution history"
        else:
            fund_basis = "Distribution-adjusted return; file completeness is unconfirmed"
except DataError as exc:
    st.error(str(exc))
    st.stop()

min_date = fund["Date"].min()
max_date = fund["Date"].max()
default_start = pd.Timestamp(year=max_date.year, month=1, day=1)
if default_start <= min_date:
    default_start = min_date + pd.Timedelta(days=1)
default_start = min(default_start, max_date)

with st.sidebar:
    selected_start = st.date_input("Analysis start date", value=default_start.date(), min_value=min_date.date(), max_value=max_date.date())
    selected_end = st.date_input("Analysis end date", value=max_date.date(), min_value=min_date.date(), max_value=max_date.date())
    st.divider()
    st.subheader("Yahoo Finance benchmark")
    mdax_ticker = st.text_input("MDAX ticker", DEFAULT_MDAX_TICKER)
    sdax_ticker = st.text_input("SDAX ticker", DEFAULT_SDAX_TICKER)
    benchmark_is_total_return = st.checkbox(
        "These Yahoo tickers are performance/total-return indices",
        value=True,
        help="Default ^MDAXI and ^SDAXI are Yahoo performance-index series. If you enter price-index symbols instead, untick this box.",
    )
    st.caption("Yahoo Finance symbols: ^MDAXI and ^SDAXI. Benchmark downloads refresh hourly.")

if selected_start > selected_end:
    st.error("The start date must be on or before the end date.")
    st.stop()

start_ts = pd.Timestamp(selected_start)
end_ts = pd.Timestamp(selected_end)
before_start = fund.loc[fund["Date"] < start_ts]
if before_start.empty:
    st.error("The fund file needs at least one closing price before the selected analysis start date.")
    st.stop()
baseline = before_start.iloc[-1]
baseline_date = pd.Timestamp(baseline["Date"])
period = fund.loc[(fund["Date"] >= baseline_date) & (fund["Date"] <= end_ts)].copy().reset_index(drop=True)
if len(period) < 2:
    st.error("The selected period needs a baseline price and at least one later observation.")
    st.stop()

# The first observation is the baseline, so its previous-period return is excluded.
period.loc[0, "Fund_Return"] = np.nan
period["Fund_Wealth"] = 100.0 * (1.0 + period["Fund_Return"].fillna(0.0)).cumprod()
fund_stats = _metrics(period["Fund_Return"], period["Fund_Wealth"])
fund_stats["return_pct"] = float(period["Fund_Wealth"].iloc[-1] - 100.0)

st.info(
    f"Fund return basis: **{fund_basis}**. Analysis baseline is the last fund close before "
    f"{start_ts:%d %b %Y}: **{baseline_date:%d %b %Y} at {baseline['Price']:.2f} EUR**. "
    f"Data ends {period['Date'].iloc[-1]:%d %b %Y}."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Fund return", _format_pct(fund_stats["return_pct"]))
col2.metric("Latest fund price", f"{period['Price'].iloc[-1]:.2f} EUR")
col3.metric("Maximum drawdown", _format_pct(fund_stats["max_drawdown_pct"]))
col4.metric("Annualized volatility", f"{fund_stats['volatility_pct']:.2f}%" if np.isfinite(fund_stats["volatility_pct"]) else "n/a")

if fund_stats["return_pct"] > 0 and fund_stats["max_drawdown_pct"] <= -10:
    interpretation = (
        "The fund finished the selected period with a positive return, but it experienced a "
        "material interim decline. The ending gain alone understates the path risk."
    )
elif fund_stats["return_pct"] > 0:
    interpretation = "The fund finished the selected period with a positive return. Review volatility and drawdown alongside the ending result."
else:
    interpretation = "The fund finished the selected period below its baseline. The size of the drawdown describes the largest interim fall from a prior peak."
st.write(interpretation)

tab_fund, tab_benchmark, tab_data = st.tabs(["Fund performance", "Benchmark comparison", "Data and methodology"])

with tab_fund:
    perf_fig = go.Figure()
    perf_fig.add_trace(go.Scatter(x=period["Date"], y=period["Fund_Wealth"], mode="lines", name="Fund", line={"color": "#117C83", "width": 3}))
    perf_fig.update_layout(title="Growth of 100 from the selected baseline", xaxis_title="Date", yaxis_title="Value (baseline = 100)", template="plotly_white", hovermode="x unified")
    st.plotly_chart(perf_fig, use_container_width=True)

    drawdown = period["Fund_Wealth"] / period["Fund_Wealth"].cummax() - 1
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=period["Date"], y=drawdown * 100, mode="lines", fill="tozeroy", name="Fund drawdown", line={"color": "#C25554", "width": 2}))
    dd_fig.update_layout(title="Fund drawdown from its prior peak", xaxis_title="Date", yaxis_title="Drawdown (%)", template="plotly_white", hovermode="x unified")
    st.plotly_chart(dd_fig, use_container_width=True)

    monthly_source = period.dropna(subset=["Fund_Return"]).copy()
    monthly = monthly_source.groupby(monthly_source["Date"].dt.to_period("M"))["Fund_Return"].apply(lambda r: (1 + r).prod() - 1)
    monthly.index = monthly.index.to_timestamp(how="end").normalize()
    if not monthly.empty:
        monthly_fig = go.Figure()
        monthly_fig.add_trace(go.Bar(x=monthly.index, y=monthly.values * 100, marker_color=["#18866B" if x >= 0 else "#C25554" for x in monthly.values], name="Monthly return"))
        monthly_fig.update_layout(title="Monthly compounded returns", xaxis_title="Month", yaxis_title="Return (%)", template="plotly_white")
        st.plotly_chart(monthly_fig, use_container_width=True)

with tab_benchmark:
    st.subheader("Deka reference index: 80% MDAX / 20% SDAX")
    st.caption("Deka states these weights for its reference index. The app forms a daily rebalanced composite from Yahoo Finance daily index closes, then normalizes both series to 100 at the same baseline.")
    benchmark_error = None
    benchmark = None
    try:
        with st.spinner("Loading MDAX and SDAX daily closes from Yahoo Finance…"):
            yahoo_levels = _download_yahoo_indices(
                baseline_date.strftime("%Y-%m-%d"),
                period["Date"].iloc[-1].strftime("%Y-%m-%d"),
                mdax_ticker,
                sdax_ticker,
            )
        benchmark = _make_benchmark(yahoo_levels, baseline_date, period["Date"])
    except Exception as exc:
        benchmark_error = str(exc)

    if benchmark_error:
        st.error(
            "A real benchmark comparison is unavailable for this run. No simulated or random benchmark is substituted. "
            f"Details: {benchmark_error}"
        )
        st.markdown(
            "Check internet access, the Yahoo ticker symbols, and whether Yahoo returned data covering the fund baseline and end date. "
            "The fund performance tab remains available."
        )
    elif (
        fund_basis.startswith("Distribution-adjusted")
        or fund_basis.startswith("Total return") != benchmark_is_total_return
    ):
        st.warning(
            "Relative performance is not calculated. Confirm that the fund distribution file covers every ex-date in the period and that the benchmark series uses the same price or total-return basis. "
            "The default Yahoo tickers are performance indices. For a total-return comparison, upload and confirm complete fund distributions. "
            "For a price-index comparison, use matching Yahoo price-index tickers and untick the total-return box."
        )
        benchmark_only = benchmark.reset_index(names="Date")
        benchmark_return_only = float(benchmark_only["Benchmark_Wealth"].iloc[-1] - 100.0)
        st.metric("80/20 benchmark return (Yahoo Finance)", _format_pct(benchmark_return_only))
        benchmark_fig = go.Figure()
        benchmark_fig.add_trace(go.Scatter(x=benchmark_only["Date"], y=benchmark_only["Benchmark_Wealth"], mode="lines", name="80% MDAX / 20% SDAX", line={"color": "#D69E2E", "width": 2}))
        benchmark_fig.update_layout(title="Benchmark series only; return basis is not comparable", xaxis_title="Date", yaxis_title="Value (baseline = 100)", template="plotly_white", hovermode="x unified")
        st.plotly_chart(benchmark_fig, use_container_width=True)
        st.caption("No fund-versus-benchmark return gap or risk comparison is shown until both series use the same price or total-return basis.")
    else:
        comparison = period[["Date", "Fund_Wealth", "Fund_Return"]].merge(
            benchmark.reset_index(names="Date"), on="Date", how="inner"
        )
        if comparison.empty or comparison["Benchmark_Wealth"].isna().any():
            st.error("Yahoo Finance data could not be aligned to the fund dates. No comparison is shown.")
        else:
            comparison.iloc[0, comparison.columns.get_loc("Benchmark_Return")] = np.nan
            benchmark_return = float(comparison["Benchmark_Wealth"].iloc[-1] - 100.0)
            spread = float(fund_stats["return_pct"] - benchmark_return)
            bdrawdown = comparison["Benchmark_Wealth"] / comparison["Benchmark_Wealth"].cummax() - 1
            bvol = comparison["Benchmark_Return"].dropna().std(ddof=1) * np.sqrt(252) * 100

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Fund return", _format_pct(fund_stats["return_pct"]))
            b2.metric("80/20 reference composite", _format_pct(benchmark_return))
            b3.metric("Return gap", f"{spread:+.2f} pp".replace(".", ","))
            b4.metric("Benchmark volatility", f"{bvol:.2f}%" if np.isfinite(bvol) else "n/a")

            relative_fig = go.Figure()
            relative_fig.add_trace(go.Scatter(x=comparison["Date"], y=comparison["Fund_Wealth"], mode="lines", name="Fund", line={"color": "#117C83", "width": 3}))
            relative_fig.add_trace(go.Scatter(x=comparison["Date"], y=comparison["Benchmark_Wealth"], mode="lines", name="80% MDAX / 20% SDAX", line={"color": "#D69E2E", "width": 2}))
            relative_fig.update_layout(title="Fund and reference composite, same dates and baseline", xaxis_title="Date", yaxis_title="Value (baseline = 100)", template="plotly_white", hovermode="x unified")
            st.plotly_chart(relative_fig, use_container_width=True)

            if spread > 0.05:
                st.write(f"Over this same period, the fund return was {spread:.2f} percentage points above the calculated reference composite.")
            elif spread < -0.05:
                st.write(f"Over this same period, the fund return was {abs(spread):.2f} percentage points below the calculated reference composite.")
            else:
                st.write("Over this same period, the fund and calculated reference composite returns were close.")
            st.caption("The return gap is a simple difference in period returns, not risk-adjusted alpha. Index daily closes are from Yahoo Finance and may be revised or incomplete.")

            bench_risk = pd.DataFrame(
                {
                    "Metric": ["Return", "Annualized volatility", "Maximum drawdown"],
                    "Fund": [fund_stats["return_pct"], fund_stats["volatility_pct"], fund_stats["max_drawdown_pct"]],
                    "80/20 reference composite": [benchmark_return, bvol, float(bdrawdown.min() * 100)],
                }
            )
            st.dataframe(bench_risk.style.format({"Fund": "{:+.2f}%", "80/20 reference composite": "{:+.2f}%"}), hide_index=True, use_container_width=True)

with tab_data:
    st.subheader("Data checks")
    st.write(f"Fund file: `{fund_filename}`")
    st.write(f"Price observations in selected period: **{len(period)}** · return intervals: **{fund_stats['return_intervals']}**")
    st.write(f"Baseline close: **{baseline_date:%Y-%m-%d} · {baseline['Price']:.2f} EUR** · Last close: **{period['Date'].iloc[-1]:%Y-%m-%d} · {period['Price'].iloc[-1]:.2f} EUR**")
    st.write(f"Return basis: **{fund_basis}**")
    st.markdown(
        "**Methodology**\n"
        "- Fund return uses consecutive fund prices. If an ex-date distribution CSV is uploaded, it adds cash distributions per share to that day's return.\n"
        "- The composite uses Yahoo Finance `^MDAXI` and `^SDAXI` closes. Daily composite return = 0.80 × MDAX return + 0.20 × SDAX return, compounded daily.\n"
        "- Both growth series start at 100 on the same last available close before the selected start date.\n"
        "- Volatility is the sample standard deviation of daily returns × √252. Drawdown is measured from the prior peak.\n"
        "- Yahoo index symbols and fund price series should use matching return conventions. The benchmark chart identifies the downloaded Yahoo series and this app's daily rebalancing assumption."
    )
    export = period[["Date", "Price", "Distribution", "Fund_Return", "Fund_Wealth"]].copy()
    st.download_button(
        "Download fund calculation table (CSV)",
        data=export.to_csv(index=False).encode("utf-8-sig"),
        file_name="deka_fund_calculations.csv",
        mime="text/csv",
    )
    st.dataframe(export.tail(20), hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "Reference-index weights: Deka-Deutschland Nebenwerte CF fund profile. "
    "Index history: Yahoo Finance via yfinance. Data availability and accuracy depend on those sources."
)
