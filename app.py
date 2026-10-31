"""Deka-Deutschland Nebenwerte CF performance dashboard.

Fondspreise: data/Trainee-Aufgabe_2026.xlsx
Referenzindex: 80 % MDAX + 20 % SDAX, anhand der Yahoo-Finance-Symbole ^MDAXI und ^SDAXI.
Die Gesamtentwicklung wird aus den täglichen Indexrenditen berechnet und täglich
auf 80/20 zurückgesetzt. Die Excel-Datei enthält nur Preise; die Fondsrendite
ist daher eine Kursrendite und berücksichtigt keine Ausschüttungen.
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


st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
        --deka-red: #C8102E;
        --deka-ink: #173A45;
        --deka-muted: #52666D;
        --deka-border: #DCE3E5;
        --deka-soft: #F5F7F7;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp,
    [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        color: var(--deka-ink) !important;
    }
    [data-testid="stHeader"] { border-bottom: 1px solid var(--deka-border); }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid var(--deka-border);
    }
    h1, h2, h3, h4 { color: var(--deka-ink) !important; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--deka-muted); }
    a { color: var(--deka-red) !important; }
    [data-testid="stMetric"] {
        background: var(--deka-soft);
        border: 1px solid var(--deka-border);
        border-top: 3px solid var(--deka-red);
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    [data-testid="stMetricValue"] { color: var(--deka-ink) !important; }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--deka-border);
        gap: 0.4rem;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] { color: var(--deka-muted); }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: var(--deka-red) !important;
        border-bottom-color: var(--deka-red) !important;
    }
    button[kind="primary"] {
        background-color: var(--deka-red) !important;
        border-color: var(--deka-red) !important;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--deka-border); border-radius: 6px; }
    hr { border-color: var(--deka-border) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


class DataError(ValueError):
    """Verständlicher Fehler bei ungültigen Quelldaten."""


def load_fund_prices(path: Path = FUND_FILE) -> pd.DataFrame:
    """Liest die Datum-/Preisreihe aus der bereitgestellten Arbeitsmappe."""
    if not path.exists():
        raise DataError(f"Excel-Datei nicht gefunden: {path}")
    try:
        raw = pd.read_excel(path, sheet_name=0, header=None)
    except Exception as exc:
        raise DataError(f"Die Excel-Datei konnte nicht gelesen werden: {exc}") from exc

    header_row = None
    for row_number, row in raw.iterrows():
        first_two = [str(value).strip().lower() for value in row.iloc[:2].tolist()]
        if any(value in {"datum", "date"} for value in first_two):
            header_row = row_number
            break
    if header_row is None:
        raise DataError("Die Spaltenüberschrift Datum/Date wurde in der Excel-Datei nicht gefunden.")

    prices = raw.iloc[header_row + 1 :, :2].copy()
    prices.columns = ["Date", "Price"]
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce", dayfirst=True)
    prices["Price"] = pd.to_numeric(prices["Price"], errors="coerce")
    prices = prices.dropna(subset=["Date", "Price"])
    prices["Date"] = prices["Date"].dt.normalize()
    prices = prices.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

    if len(prices) < 2:
        raise DataError("Die Excel-Datei muss mindestens zwei gültige Fondspreise enthalten.")
    if (prices["Price"] <= 0).any():
        raise DataError("Fondspreise müssen größer als null sein.")
    return prices


@st.cache_data(ttl=60 * 60, show_spinner=False)
def download_yahoo_indices(start_date: str, end_date_exclusive: str) -> pd.DataFrame:
    """Lädt tägliche Yahoo-Schlusskurse für den Zeitraum der Excel-Daten herunter (Enddatum exklusiv)."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Das Paket yfinance fehlt. Bitte in requirements.txt ergänzen und installieren.") from exc

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
        raise RuntimeError(f"Der Yahoo-Finance-Abruf ist fehlgeschlagen: {exc}") from exc
    if data is None or data.empty:
        raise RuntimeError("Yahoo Finance hat für den Zeitraum der Excel-Daten keine Indexdaten geliefert.")

    def close_series(ticker: str, output_name: str) -> pd.Series:
        matches = []
        for column in data.columns:
            parts = column if isinstance(column, tuple) else (column,)
            normalised = {str(part).strip().casefold() for part in parts}
            if ticker.casefold() in normalised and "close" in normalised:
                matches.append(column)
        if len(matches) != 1:
            raise RuntimeError(f"In der Yahoo-Antwort wurde keine eindeutige tägliche Schlusskursreihe für {ticker} gefunden.")
        series = pd.to_numeric(data[matches[0]], errors="coerce")
        series.index = pd.to_datetime(series.index, utc=True).tz_convert(None).normalize()
        series = series[~series.index.duplicated(keep="last")].sort_index()
        series.name = output_name
        return series

    indices = pd.concat(
        [close_series(MDAX_TICKER, "MDAX"), close_series(SDAX_TICKER, "SDAX")], axis=1
    ).sort_index()
    if indices.empty or (indices <= 0).any().any():
        raise RuntimeError("Die Yahoo-Finance-Schlusskurse sind leer oder ungültig.")
    return indices


def make_benchmark(indices: pd.DataFrame, fund_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Berechnet die täglich auf 80/20 zurückgesetzte Referenzindex und übernimmt Werte zu den Excel-Stichtagen."""
    daily = indices.loc[(indices.index >= fund_dates[0]) & (indices.index <= fund_dates[-1])].copy()
    missing = daily[daily.isna().any(axis=1)].index
    if len(missing):
        examples = ", ".join(date.strftime("%Y-%m-%d") for date in missing[:5])
        raise RuntimeError(
            f"Für {len(missing)} Handelstag(e) fehlen MDAX- oder SDAX-Schlusskurse bei Yahoo, darunter {examples}. "
            "Fehlende Referenzindexwerte werden nicht fortgeschrieben."
        )

    missing_excel_dates = fund_dates.difference(daily.index)
    if len(missing_excel_dates):
        examples = ", ".join(date.strftime("%Y-%m-%d") for date in missing_excel_dates[:5])
        raise RuntimeError(f"Für {len(missing_excel_dates)} Excel-Beobachtungstag(e) liegen keine Indexschlusskurse von Yahoo vor, darunter {examples}.")
    if len(daily) < 2 or daily.index[0] != fund_dates[0]:
        raise RuntimeError("Die Yahoo-Daten enthalten nicht den ersten Excel-Beobachtungstag als Referenzindexbasis.")

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
    """Gibt das Jahr einschließlich des letzten vorherigen Beobachtungstags als Renditebasis zurück."""
    in_year = prices.index[prices["Date"].dt.year.eq(year)]
    if len(in_year) == 0:
        raise DataError(f"Die Excel-Datei enthält keine Fondspreise für {year}.")
    first_year_idx = int(in_year[0])
    last_year_idx = int(in_year[-1])
    has_prior_base = first_year_idx > 0
    start_idx = first_year_idx - 1 if has_prior_base else first_year_idx
    return prices.loc[start_idx:last_year_idx].copy().reset_index(drop=True), has_prior_base


GERMAN_MONTHS = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"]

def de_date(value) -> str:
    return pd.Timestamp(value).strftime("%d.%m.%Y")


def de_num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "k. A."
    return f"{value:+.2f}%".replace(".", ",")
def style_deka_chart(figure: go.Figure) -> go.Figure:
    """Apply the light Deka palette to Plotly's canvas and chart furniture."""
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family": "Arial, sans-serif", "color": "#173A45"},
        title={"font": {"color": "#173A45"}},
        legend={"font": {"color": "#173A45"}, "bgcolor": "rgba(255,255,255,0)"},
        hoverlabel={"bgcolor": "#FFFFFF", "font": {"color": "#173A45"}, "bordercolor": "#DCE3E5"},
        xaxis={
            "tickfont": {"color": "#52666D"},
            "title": {"font": {"color": "#173A45"}},
            "gridcolor": "#E6EBED",
            "linecolor": "#BCC8CC",
            "zerolinecolor": "#BCC8CC",
            "showline": True,
        },
        yaxis={
            "tickfont": {"color": "#52666D"},
            "title": {"font": {"color": "#173A45"}},
            "gridcolor": "#E6EBED",
            "linecolor": "#BCC8CC",
            "zerolinecolor": "#BCC8CC",
            "showline": True,
        },
    )
    return figure


def style_deka_table(frame: pd.DataFrame):
    """Force table cells and headers onto the same light Deka palette."""
    styler = frame.style.set_properties(
        **{
            "background-color": "#FFFFFF",
            "color": "#173A45",
            "border-color": "#DCE3E5",
        }
    )
    return styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#EAF0F1"),
                    ("color", "#173A45"),
                    ("font-weight", "bold"),
                    ("border-color", "#DCE3E5"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", "#FFFFFF"),
                    ("color", "#173A45"),
                    ("border-color", "#DCE3E5"),
                ],
            },
        ],
        overwrite=False,
    )


st.title(f"📈 {FUND_NAME}")
st.caption(f"ISIN {ISIN} · Fondspreise aus der bereitgestellten Excel-Datei · Referenzindex: 80 % MDAX / 20 % SDAX")

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
    f"Bewertungszeitraum bis zur letzten Excel-Beobachtung: **{de_date(end_date)}** · "
    f"{len(fund)} Beobachtungen einschließlich Jahresanfangs-Basis · {fund_stats['return_intervals']} Renditeintervalle. "
    + ("Die Rendite seit Jahresbeginn beginnt mit dem letzten verfügbaren Fondspreis aus 2025." if has_ytd_baseline else "Es liegt kein Fondspreis aus 2025 vor. Daher kann die genaue Rendite seit dem Jahresende 2025 nicht berechnet werden.")
    + " Die Excel-Datei enthält nur Fondspreise. Die Fondsentwicklung ist daher eine Kursrendite ohne Ausschüttungen."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Kursrendite seit Jahresbeginn 2026", pct(fund_stats["return_pct"]), help="Berechnet vom letzten verfügbaren Fondspreis 2025 bis zum letzten Excel-Preis 2026.")
c2.metric("Letzter Fondspreis", f"{fund['Price'].iloc[-1]:.2f} EUR".replace(".", ","))
c3.metric("Maximaler Verlust 2026", pct(fund_stats["max_drawdown_pct"]))
c4.metric("Volatilität p. a.", f"{fund_stats['volatility_pct']:.2f}%".replace(".", ",") if np.isfinite(fund_stats["volatility_pct"]) else "k. A.")

direction = "positiv" if fund_stats["return_pct"] > 0 else "negativ" if fund_stats["return_pct"] < 0 else "unverändert"
sortino_summary = f"{fund_stats['sortino_ratio']:.2f}".replace(".", ",") if np.isfinite(fund_stats["sortino_ratio"]) else "k. A."
st.write(
    f"Bis zum {de_date(end_date)} beträgt die Kursrendite des Fonds im Jahr 2026 {pct(fund_stats['return_pct'])} ({direction}). "
    f"Die Volatilität p. a. beträgt {de_num(fund_stats['volatility_pct'], 1)} %, der maximale Verlust (Drawdown) {pct(fund_stats['max_drawdown_pct'])} "
    f"vom {de_date(fund_stats['peak_date'])} bis zum {de_date(fund_stats['trough_date'])}. "
    f"Die Sortino-Ratio beträgt {sortino_summary} bei einem Zielwert von 0 %. Rendite und Risikoverlauf sollten gemeinsam beurteilt werden. "
    "Im Reiter Referenzindexvergleich wird der Fonds an denselben Stichtagen mit dem 80/20-MDAX/SDAX-Index verglichen; dabei ist die unterschiedliche Renditebasis zu beachten."
)

profile_tab, fund_tab, benchmark_tab, data_tab = st.tabs(["Fondsprofil", "Fondsentwicklung", "Referenzindexvergleich", "Daten und Methodik"])

with profile_tab:
    st.subheader("Fondsprofil und Portfoliostand")
    st.caption(f"Aus der bereitgestellten Deka-Präsentation übernommen · Zusammensetzung und Bewertungen zum {PROFILE_AS_OF}")
    st.subheader("Fondsratings")
    rating_left, rating_right = st.columns(2)
    rating_left.metric("Morningstar", "★★★☆☆ · 3 von 5 Sternen")
    rating_right.metric("Scope-Fondsrating", "C")
    st.caption(f"Ratingstand laut bereitgestellter Deka-Präsentation vom {PROFILE_AS_OF}. Bewertungen sind stichtagsbezogen und können sich ändern.")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Fondsauflegung", "03.06.2013")
    p2.metric("Ertragsverwendung", "Ausschüttend")
    p3.metric("Mindestanlage", "€25")
    p4.metric("Geschäftsjahresende", "31. Mai")

    left, right = st.columns(2)
    with left:
        industry_frame = pd.DataFrame({"Branche": list(INDUSTRIES), "Anteil (%)": list(INDUSTRIES.values())}).sort_values("Anteil (%)")
        industry_chart = go.Figure(go.Bar(x=industry_frame["Anteil (%)"], y=industry_frame["Branche"], orientation="h", marker_color="#C8102E", text=industry_frame["Anteil (%)"].map(lambda x: f"{x:.1f}%".replace(".", ",")), textposition="outside"))
        industry_chart.update_layout(title=f"Größte Branchen · {PROFILE_AS_OF}", xaxis_title="Portfolioanteil (%)", yaxis_title="", template="plotly_white", margin={"l": 10, "r": 40, "t": 55, "b": 35})
        st.plotly_chart(style_deka_chart(industry_chart), use_container_width=True)
    with right:
        holding_frame = pd.DataFrame({"Wertpapier": list(TOP_HOLDINGS), "Anteil (%)": list(TOP_HOLDINGS.values())}).sort_values("Anteil (%)")
        holdings_chart = go.Figure(go.Bar(x=holding_frame["Anteil (%)"], y=holding_frame["Wertpapier"], orientation="h", marker_color="#527D89", text=holding_frame["Anteil (%)"].map(lambda x: f"{x:.1f}%".replace(".", ",")), textposition="outside"))
        holdings_chart.update_layout(title=f"Zehn größte Positionen · {PROFILE_AS_OF}", xaxis_title="Portfolioanteil (%)", yaxis_title="", template="plotly_white", margin={"l": 10, "r": 40, "t": 55, "b": 35})
        st.plotly_chart(style_deka_chart(holdings_chart), use_container_width=True)

    top_sector_total = sum(INDUSTRIES.values())
    top_holding_total = sum(TOP_HOLDINGS.values())
    top_sector_share = de_num(top_sector_total, 1)
    st.write(f"Die zehn größten dargestellten Branchen machen **{top_sector_share}%** des Portfolios aus; die zehn größten Positionen zusammen **{de_num(top_holding_total, 1)}%**. Die Branchengrafik zeigt nur die größten Gruppen, nicht die vollständige Portfolioaufteilung.")
    st.info("Dies sind statische Angaben aus der Präsentation mit Stand 30. September 2026. Sie dienen als Kontext zur Wertentwicklung im Excel-Zeitraum und sind weder als aktuelle Bestände noch als Bestände an jedem Beobachtungstag zu verstehen.")

    with st.expander("Kosten und unabhängige Bewertungen laut Präsentation"):
        st.markdown(
            """- Jährliche Verwaltungsvergütung: **1,86 %**; Transaktionskosten: **0,40 %**; erfolgsabhängige Vergütung: **0,41 %** unter bestimmten Voraussetzungen (Kostenstand der Präsentation: 01.10.2026).
- Die Präsentation nennt in ihrer Erläuterung zur Wertentwicklung einen maximalen Ausgabeaufschlag von **3,610 %**. Einstiegskosten hängen vom Vertriebskanal ab und sind nicht in der hier berechneten Fondskursrendite enthalten.
- Ratings zum 30.09.2026: Morningstar **3/5 Sterne**; Scope **C**. Dies sind die in der Präsentation gezeigten Werte, keine aktuelle Ratingabfrage."""
        )

with fund_tab:
    performance = go.Figure()
    performance.add_trace(go.Scatter(x=fund["Date"], y=fund["Price"], mode="lines", name="Fondspreis (EUR)", line={"color": "#C8102E", "width": 3}, hovertemplate="%{x|%d.%m.%Y}<br>Kurs: %{y:.2f} EUR<extra></extra>"))
    performance.update_layout(title="Entwicklung des Fondspreises", xaxis_title="Datum", yaxis_title="Fondspreis (EUR)", template="plotly_white", hovermode="x unified", xaxis_tickformat="%d.%m.%Y")
    st.plotly_chart(style_deka_chart(performance), use_container_width=True)

    drawdown = fund["Fund_Wealth"] / fund["Fund_Wealth"].cummax() - 1.0
    dd_chart = go.Figure()
    dd_chart.add_trace(go.Scatter(x=fund["Date"], y=drawdown * 100.0, mode="lines", fill="tozeroy", name="Kursrückgang", line={"color": "#C8102E", "width": 2}))
    dd_chart.update_layout(title="Kursrückgang vom vorherigen Höchststand", xaxis_title="Datum", yaxis_title="Kursrückgang (%)", template="plotly_white", hovermode="x unified", xaxis_tickformat="%d.%m.%Y")
    st.plotly_chart(style_deka_chart(dd_chart), use_container_width=True)

    r1, r2, r3 = st.columns(3)
    r1.metric("Aktueller Rückgang vom Höchststand", pct(fund_stats["current_drawdown_pct"]))
    r2.metric("Höchststand vor dem maximalen Rückgang", de_date(fund_stats["peak_date"]))
    r3.metric("Tiefpunkt des maximalen Rückgangs", de_date(fund_stats["trough_date"]))
    if fund_stats["recovery_date"] is not None:
        st.write(f"Der Fonds erreichte seinen vorherigen Höchststand am **{de_date(fund_stats['recovery_date'])}** wieder, {fund_stats['recovery_days']} Kalendertage nach dem Tiefpunkt.")
    else:
        st.write(f"Bis zum **{de_date(end_date)}** hatte der Fonds den vorherigen Höchststand noch nicht wieder erreicht; seit dem Höchststand lag er {fund_stats['underwater_days']} Kalendertage darunter.")

    risk1, risk2, risk3, risk4, risk5 = st.columns(5)
    risk1.metric("Abwärtsabweichung p. a.", f"{fund_stats['downside_deviation_pct']:.2f}%".replace(".", ",") if np.isfinite(fund_stats["downside_deviation_pct"]) else "k. A.")
    risk2.metric("Sortino-Ratio · 0 %-Ziel", f"{fund_stats['sortino_ratio']:.2f}".replace(".", ",") if np.isfinite(fund_stats["sortino_ratio"]) else "k. A.")
    risk3.metric("Historischer Tages-VaR · 95 %", pct(fund_stats["var_95_pct"]))
    risk4.metric("Historischer Tages-CVaR · 95 %", pct(fund_stats["cvar_95_pct"]))
    risk5.metric("Anteil positiver Beobachtungen", f"{fund_stats['positive_days_pct']:.1f}%".replace(".", ",") if np.isfinite(fund_stats["positive_days_pct"]) else "k. A.")
    st.caption("Abwärtsabweichung und Sortino-Ratio verwenden ein tägliches Ziel von 0 % und werden mit 252 Handelstagen annualisiert. Der historische VaR entspricht dem 5. Perzentil der beobachteten Tagesrenditen; der CVaR ist die durchschnittliche Rendite im schlechtesten 5-%-Randbereich. Dies sind rückblickende Schätzwerte, keine Verlustgrenzen oder Prognosen.")

    best_date = fund_dates_indexed["Fund_Return"].idxmax()
    worst_date = fund_dates_indexed["Fund_Return"].idxmin()
    best, worst = st.columns(2)
    best.metric("Bester beobachteter Tageswert", pct(fund_stats["best_day_pct"]), help=f"Beobachtungstag: {de_date(best_date)}")
    worst.metric("Schlechtester beobachteter Tageswert", pct(fund_stats["worst_day_pct"]), help=f"Beobachtungstag: {de_date(worst_date)}")

    monthly_source = fund.dropna(subset=["Fund_Return"]).copy()
    monthly = monthly_source.groupby(monthly_source["Date"].dt.to_period("M"))["Fund_Return"].apply(lambda values: (1.0 + values).prod() - 1.0)
    monthly.index = monthly.index.to_timestamp(how="end").normalize()
    monthly_chart = go.Figure()
    monthly_chart.add_trace(go.Bar(x=monthly.index, y=monthly.values * 100.0, marker_color=["#18866B" if value >= 0 else "#C8102E" for value in monthly.values], name="Monatliche Kursrendite"))
    monthly_chart.update_layout(title="Aufgezinste monatliche Fondskursrenditen", xaxis_title="Monat", yaxis_title="Rendite (%)", template="plotly_white", xaxis_tickformat="%m.%Y")
    st.plotly_chart(style_deka_chart(monthly_chart), use_container_width=True)
    monthly_positive = int((monthly > 0).sum())
    st.caption(f"Monate mit positiver Rendite: {monthly_positive} von {len(monthly)}. Der erste Monatswert beginnt mit dem letzten verfügbaren Fondspreis 2025; der letzte Monat endet mit der jüngsten Excel-Beobachtung.")

with benchmark_tab:
    st.subheader("Deka-Referenzindex: 80 % MDAX / 20 % SDAX")
    st.caption("Tägliche Schlusskurse von Yahoo Finance für ^MDAXI und ^SDAXI. Die Kombination wird als 80 % × tägliche MDAX-Rendite + 20 % × tägliche SDAX-Rendite berechnet und an jedem Handelstag auf diese Gewichte zurückgesetzt.")
    benchmark_error = None
    benchmark = None
    try:
        end_exclusive = (end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        with st.spinner("Yahoo-Finance-Indexdaten für die Excel-Beobachtungstage werden geladen …"):
            index_data = download_yahoo_indices(start_date.strftime("%Y-%m-%d"), end_exclusive)
        benchmark = make_benchmark(index_data, fund_dates)
    except Exception as exc:
        benchmark_error = str(exc)

    if benchmark_error:
        st.error(
            "Ein Vergleich mit dem Referenzindex zu denselben Stichtagen ist nicht verfügbar. Es werden keine Ersatzwerte oder Ersatzstichtage verwendet. "
            f"Fehlerdetails: {benchmark_error}"
        )
    else:
        fund_return = fund_stats["return_pct"]
        benchmark_return = float(benchmark["Benchmark_Wealth"].iloc[-1] - 100.0)
        gap_pp = fund_return - benchmark_return
        benchmark_stats = calculate_metrics(benchmark["Benchmark_Return"], benchmark["Benchmark_Wealth"])

        st.warning(
            "Hinweis zur Renditebasis: Die Excel-Datei enthält nur Fondspreise; die MDAX-/SDAX-Reihen von Yahoo sind Performanceindizes. "
            "Der Referenzindex berücksichtigt Dividenden, die Fondskursrendite hier nicht. Die Differenz ist deshalb kein Vergleich auf gleicher Gesamtrenditebasis und kein Alpha."
        )
        st.warning(
            "Diese App berechnet die 80/20-Kombination mit täglicher Anpassung der Gewichte. Deka veröffentlicht die Indexgewichte; die Excel-Datei enthält jedoch weder eine offizielle zusammengesetzte Renditereihe noch deren detaillierte Berechnungsmethode."
        )

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Fondskursrendite", pct(fund_return))
        b2.metric("Yahoo-Referenzindexrendite 80/20", pct(benchmark_return))
        b3.metric("Abweichung (Prozentpunkte)", f"{gap_pp:+.2f} Prozentpunkte".replace(".", ","))
        b4.metric("Volatilität des Referenzindex p. a.", f"{benchmark_stats['volatility_pct']:.2f}%".replace(".", ",") if np.isfinite(benchmark_stats["volatility_pct"]) else "k. A.")

        comparison = go.Figure()
        comparison.add_trace(go.Scatter(x=fund["Date"], y=fund["Fund_Wealth"], mode="lines", name="Fondskursrendite", line={"color": "#C8102E", "width": 3}))
        comparison.add_trace(go.Scatter(x=benchmark.index, y=benchmark["Benchmark_Wealth"], mode="lines", name="80 % MDAX / 20 % SDAX", line={"color": "#527D89", "width": 2}))
        comparison.update_layout(title="Fonds und Referenzindex an identischen Excel-Stichtagen", xaxis_title="Datum", yaxis_title="Entwicklung (erster Excel-Stichtag = 100)", template="plotly_white", hovermode="x unified")
        st.plotly_chart(style_deka_chart(comparison), use_container_width=True)
        st.subheader("Wertentwicklung")
        st.caption("Fondskursrendite aus der Excel-Reihe und Rendite des 80/20-MDAX-/SDAX-Referenzindex an denselben Excel-Beobachtungstagen.")

        fund_wealth_by_date = fund.set_index("Date")["Fund_Wealth"]
        benchmark_wealth_by_date = benchmark["Benchmark_Wealth"]
        period_rows = []

        def add_return_period(label: str, baseline_date: pd.Timestamp, period_end: pd.Timestamp) -> None:
            fund_period_return = (fund_wealth_by_date.loc[period_end] / fund_wealth_by_date.loc[baseline_date] - 1.0) * 100.0
            benchmark_period_return = (benchmark_wealth_by_date.loc[period_end] / benchmark_wealth_by_date.loc[baseline_date] - 1.0) * 100.0
            period_rows.append({
                "Zeitraum": label,
                "Fonds (Excel-Kursrendite)": f"{fund_period_return:+.2f}%".replace(".", ","),
                "Referenzindex (80/20)": f"{benchmark_period_return:+.2f}%".replace(".", ","),
                "Abweichung (Prozentpunkte)": f"{fund_period_return - benchmark_period_return:+.2f} Prozentpunkte".replace(".", ","),
            })

        year_observations = fund.loc[fund["Date"].dt.year.eq(2026), "Date"]
        if not year_observations.empty:
            ytd_start = fund_dates[0] if has_ytd_baseline else year_observations.iloc[0]
            ytd_end = year_observations.iloc[-1]
            add_return_period(f"Seit Jahresbeginn 2026 · {ytd_start:%d.%m.%Y}–{ytd_end:%d.%m.%Y}", ytd_start, ytd_end)

            monthly_dates = fund.loc[fund["Date"].dt.year.eq(2026), ["Date"]].copy()
            monthly_dates["Month"] = monthly_dates["Date"].dt.to_period("M")
            for month in sorted(monthly_dates["Month"].unique(), reverse=True):
                month_observations = monthly_dates.loc[monthly_dates["Month"].eq(month), "Date"]
                end_of_month = month_observations.iloc[-1]
                first_position = fund_dates.get_loc(month_observations.iloc[0])
                if first_position == 0:
                    continue
                baseline = fund_dates[first_position - 1]
                add_return_period(f"{GERMAN_MONTHS[month.month - 1]} {month.year} · {baseline:%d.%m.%Y}–{end_of_month:%d.%m.%Y}", baseline, end_of_month)

        if period_rows:
            returns_table = pd.DataFrame(period_rows)

            def return_color(value: str) -> str:
                if value.startswith("+"):
                    return "color: #137A67; font-weight: 600"
                if value.startswith("-"):
                    return "color: #C8102E; font-weight: 600"
                return ""

            style_subset = ["Fonds (Excel-Kursrendite)", "Referenzindex (80/20)", "Abweichung (Prozentpunkte)"]
            styler = style_deka_table(returns_table)
            style_map = getattr(styler, "map", None)
            if style_map is not None:
                styled_returns = style_map(return_color, subset=style_subset)
            else:
                styled_returns = styler.applymap(return_color, subset=style_subset)
            st.dataframe(styled_returns, hide_index=True, use_container_width=True)
        st.caption("Eine getrennte Brutto- und Nettowertentwicklung lässt sich aus der Excel-Datei nicht ermitteln: Sie enthält nur eine Fondspreisreihe, aber keine gesonderten Angaben zu Anlegerkosten. Die Tabelle zeigt daher die beobachtete Kursrendite; sie ist keine BVI-Bruttowertentwicklung und enthält keine individuelle Anlegerkostenberechnung.")

        pct_metrics = ["Rendite", "Volatilität p. a.", "Abwärtsabweichung p. a.", "Maximaler Verlust (Drawdown)", "Aktueller Abstand zum Höchststand"]
        fund_risk_values = [fund_stats["return_pct"], fund_stats["volatility_pct"], fund_stats["downside_deviation_pct"], fund_stats["sortino_ratio"], fund_stats["max_drawdown_pct"], fund_stats["current_drawdown_pct"]]
        benchmark_risk_values = [benchmark_stats["return_pct"], benchmark_stats["volatility_pct"], benchmark_stats["downside_deviation_pct"], benchmark_stats["sortino_ratio"], benchmark_stats["max_drawdown_pct"], benchmark_stats["current_drawdown_pct"]]
        risk_labels = ["Rendite", "Volatilität p. a.", "Abwärtsabweichung p. a.", "Sortino-Ratio (0 %-Ziel)", "Maximaler Verlust (Drawdown)", "Aktueller Abstand zum Höchststand"]
        format_risk = lambda values: ["k. A." if not np.isfinite(value) else (f"{value:+.2f}%".replace(".", ",") if label in pct_metrics else f"{value:.2f}".replace(".", ",")) for label, value in zip(risk_labels, values)]
        risk_table = pd.DataFrame({"Kennzahl": risk_labels, "Fonds (Excel-Kursreihe)": format_risk(fund_risk_values), "Yahoo-Referenzindex 80/20": format_risk(benchmark_risk_values)})
        st.dataframe(style_deka_table(risk_table), hide_index=True, use_container_width=True)

        paired_returns = pd.concat(
            [fund_dates_indexed["Fund_Return"].rename("Fund"), benchmark["Benchmark_Return"].rename("Referenzindex")], axis=1
        ).dropna()
        if len(paired_returns) > 2 and paired_returns["Referenzindex"].var(ddof=1) > 0:
            active_returns = paired_returns["Fund"] - paired_returns["Referenzindex"]
            beta = float(paired_returns["Fund"].cov(paired_returns["Referenzindex"]) / paired_returns["Referenzindex"].var(ddof=1))
            correlation = float(paired_returns["Fund"].corr(paired_returns["Referenzindex"]))
            tracking_error = float(active_returns.std(ddof=1) * np.sqrt(252) * 100.0)
            information_ratio = float(active_returns.mean() * 252 / (tracking_error / 100.0)) if tracking_error > 0 else float("nan")
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Renditekorrelation", f"{correlation:.2f}".replace(".", ","))
            a2.metric("Beta zum 80/20-Referenzindex", f"{beta:.2f}".replace(".", ","))
            a3.metric("Nachbildungsfehler p. a.", f"{tracking_error:.2f}%".replace(".", ","))
            a4.metric("Information Ratio", f"{information_ratio:.2f}".replace(".", ",") if np.isfinite(information_ratio) else "k. A.")
            st.caption("Beta, Korrelation, Nachbildungsfehler und Information Ratio verwenden Renditen zu identischen Excel-Stichtagen. Da Fondskursrendite und Yahoo-Performanceindex Dividenden unterschiedlich behandeln, sind die aktiven Risikokennzahlen nur indikativ und kein direkter Leistungsvergleich.")
        if gap_pp > 0.05:
            st.write(f"An denselben Excel-Stichtagen lag die Fondskursrendite um {gap_pp:.2f} Prozentpunkte über der berechneten 80/20-Referenzindex. Wegen der unterschiedlichen Renditebasis ist dies nur als Richtungshinweis zu verstehen.")
        elif gap_pp < -0.05:
            st.write(f"An denselben Excel-Stichtagen lag die Fondskursrendite um {abs(gap_pp):.2f} Prozentpunkte unter der berechneten 80/20-Referenzindex. Wegen der unterschiedlichen Renditebasis ist dies nur als Richtungshinweis zu verstehen.")
        else:
            st.write("An denselben Excel-Stichtagen lagen Fondskursrendite und berechnete 80/20-Referenzindex nahe beieinander. Die Renditegrundlagen unterscheiden sich weiterhin.")

with data_tab:
    st.subheader("Quelldaten")
    st.write(f"Fondsquelle: `{FUND_FILE.name}`")
    st.write(f"Fondsbeobachtungen: **{len(fund)}**, vom **{start_date:%d.%m.%Y}** bis **{end_date:%d.%m.%Y}**.")
    st.write("Referenzindexquelle: Yahoo Finance über `yfinance`, Symbole `^MDAXI` und `^SDAXI`.")
    st.write("Deka-Referenzgewichte: 80 % MDAX und 20 % SDAX.")
    st.markdown(
        "**Berechnungen**\n"
        "- Kursrendite seit Jahresbeginn 2026 = letzter Excel-Fondspreis 2026 / letzter Excel-Fondspreis 2025 − 1 (letzte Beobachtung 2025 als Basis).\n"
        "- Tägliche Fondsrendite = heutiger Excel-Fondspreis / vorheriger Excel-Fondspreis − 1.\n"
        "- Tägliche Referenzindexrendite = 0,80 × tägliche MDAX-Rendite + 0,20 × tägliche SDAX-Rendite; der Referenzindex wird täglich aufgezinzt und zu den Excel-Stichtagen ausgewiesen.\n"
        "- Beide Vergleichsreihen beginnen am ersten Excel-Stichtag bei 100 und enden am letzten Excel-Stichtag.\n"
        "- Volatilität p. a. = Stichprobenstandardabweichung der Tagesrenditen × √252.\n"
        "- Maximaler Drawdown = niedrigster Tageswert im Verhältnis zum vorherigen Höchststand.\n"
        "- Abwärtsabweichung und Sortino-Ratio verwenden ein tägliches Ziel von 0 %; der historische VaR/CVaR basiert auf den schlechtesten beobachteten 5 %.\n"
        "- Die Renditedifferenz wird in Prozentpunkten gemessen; sie ist kein risikobereinigtes Alpha."
    )
    export = fund[["Date", "Price", "Fund_Return", "Fund_Wealth"]].copy()
    export.columns = ["Datum", "Fondspreis (EUR)", "Tagesrendite", "Wertentwicklung (Basis 100)"]
    st.download_button("Fondsberechnungstabelle herunterladen", export.to_csv(index=False).encode("utf-8-sig"), file_name="deka_fondsberechnungen.csv", mime="text/csv")
    st.dataframe(style_deka_table(export.tail(20)), hide_index=True, use_container_width=True)

st.divider()
st.caption("Fondsdaten werden ausschließlich aus der bereitgestellten Excel-Datei gelesen. Daten des Referenzindex werden über Yahoo Finance abgerufen und eine Stunde zwischengespeichert. Verfügbarkeit und Historie der Yahoo-Daten können sich ändern.")

