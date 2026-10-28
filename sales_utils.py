"""Data loading, cleaning, filtering and aggregation helpers for the Regional Sales app.

Everything in this module is plain pandas/numpy (no Streamlit / Plotly), so it can be
unit-tested and reused from notebooks or scripts.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Paths / schema
# --------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXCEL_PATH = os.path.join(DATA_DIR, "Denormalize_regional_sales_data.xlsx")
CACHE_PATH = os.path.join(DATA_DIR, "sales_clean.csv.gz")

RENAME = {
    "Index": "OrderID",
    "Units Sold": "Units",
    "Sales": "Revenue",
    "Cost of sales": "Cost",
    "Sale price": "UnitPrice",
    "Cost price": "UnitCost",
    "Product Name": "Product",
    "Cities": "City",
    "SalesPerson Name": "SalesPerson",
    "Scenario Name": "Scenario",
}
REQUIRED = [
    "OrderID", "Date", "Units", "Revenue", "Cost", "Product", "Category", "Brand",
    "Warranty", "City", "SalesPerson", "Designation", "Scenario",
]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _read_raw_excel(source) -> pd.DataFrame:
    """Read the workbook and auto-detect the header row (the file has blank rows on top)."""
    raw = pd.read_excel(source, header=None, engine="openpyxl")
    raw = raw.dropna(how="all").reset_index(drop=True)
    # header row = first row whose cells contain 'Date' and 'Sales' (or 'Revenue')
    header_idx = None
    for i in range(min(len(raw), 20)):
        vals = {str(v).strip() for v in raw.iloc[i].tolist()}
        if "Date" in vals and ("Sales" in vals or "Revenue" in vals):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find a header row containing 'Date' and 'Sales' columns.")
    header = [str(c).strip() for c in raw.iloc[header_idx].tolist()]
    df = raw.iloc[header_idx + 1:].copy()
    df.columns = header
    return df.reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, fix dtypes, drop unusable rows and add derived columns."""
    df = df.rename(columns=RENAME).copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "DOJ" in df.columns:
        df["DOJ"] = pd.to_datetime(df["DOJ"], errors="coerce")
    for c in ["OrderID", "Units", "Revenue", "Cost", "UnitPrice", "UnitCost", "ProductID",
              "SalesrepID", "CityID", "ScenarioID", "Age", "Latitude", "Longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["Product", "Category", "Brand", "Warranty", "City", "SalesPerson", "Designation", "Scenario"]:
        df[c] = df[c].astype(str).str.strip()

    df = df.dropna(subset=["Date", "Revenue", "Cost", "Units"]).drop_duplicates()
    df["Profit"] = df["Revenue"] - df["Cost"]
    df["Year"] = df["Date"].dt.year.astype(int)
    df["Quarter"] = "Q" + df["Date"].dt.quarter.astype(str)
    df["MonthNum"] = df["Date"].dt.month.astype(int)
    df["MonthName"] = df["MonthNum"].map(lambda m: MONTH_NAMES[m - 1])
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    return df.sort_values("Date").reset_index(drop=True)


def load_data(source=None, use_cache: bool = True) -> pd.DataFrame:
    """Load the sales data.

    * ``source=None``   -> use the bundled dataset (fast CSV cache if present, else the Excel file)
    * ``source=<file>`` -> an uploaded Excel (.xlsx) / CSV file-like object or path with the same layout
    """
    if source is None:
        if use_cache and os.path.exists(CACHE_PATH):
            df = pd.read_csv(CACHE_PATH, parse_dates=["Date", "DOJ"])
            return clean(df)
        df = clean(_read_raw_excel(EXCEL_PATH))
        try:  # best-effort cache so later starts take < 1 second and need no openpyxl parse
            slim = df.drop(columns=["Profit", "Year", "Quarter", "MonthNum", "MonthName", "Month"])
            slim.to_csv(CACHE_PATH, index=False, compression="gzip")
        except OSError:
            pass
        return df

    name = getattr(source, "name", str(source)).lower()
    if name.endswith(".csv"):
        raw = pd.read_csv(source)
    else:
        if hasattr(source, "read"):
            source = io.BytesIO(source.read())
        raw = _read_raw_excel(source)
    return clean(raw)


# --------------------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------------------
@dataclass
class Filters:
    start: pd.Timestamp
    end: pd.Timestamp
    cities: Sequence[str]
    categories: Sequence[str]
    brands: Sequence[str]
    warranties: Sequence[str]
    designations: Sequence[str]
    salespeople: Sequence[str]

    def describe(self) -> str:
        return (f"{self.start:%d %b %Y} – {self.end:%d %b %Y} | "
                f"{len(self.cities)} cities | {len(self.categories)} categories | "
                f"{len(self.brands)} brands | {len(self.salespeople)} sales reps")


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    mask = (
        (df["Date"] >= pd.Timestamp(f.start))
        & (df["Date"] <= pd.Timestamp(f.end))
        & df["City"].isin(f.cities)
        & df["Category"].isin(f.categories)
        & df["Brand"].isin(f.brands)
        & df["Warranty"].isin(f.warranties)
        & df["Designation"].isin(f.designations)
        & df["SalesPerson"].isin(f.salespeople)
    )
    return df.loc[mask]


# --------------------------------------------------------------------------------------
# Aggregations
# --------------------------------------------------------------------------------------
def _safe_div(a, b):
    return np.where(np.asarray(b, dtype=float) == 0, np.nan, np.asarray(a, dtype=float) / np.asarray(b, dtype=float))


def summarize(df: pd.DataFrame, by: str | Sequence[str]) -> pd.DataFrame:
    """Revenue / Cost / Profit / Units / Orders / Margin % / Avg order value grouped by ``by``."""
    by = [by] if isinstance(by, str) else list(by)
    out = (
        df.groupby(by, as_index=False)
        .agg(Revenue=("Revenue", "sum"), Cost=("Cost", "sum"), Profit=("Profit", "sum"),
             Units=("Units", "sum"), Orders=("OrderID", "count"))
    )
    out["Margin %"] = _safe_div(out["Profit"], out["Revenue"]) * 100
    out["Avg Order Value"] = _safe_div(out["Revenue"], out["Orders"])
    return out


def kpis(df: pd.DataFrame) -> dict:
    revenue = float(df["Revenue"].sum())
    cost = float(df["Cost"].sum())
    profit = revenue - cost
    orders = int(len(df))
    units = int(df["Units"].sum())
    return {
        "Revenue": revenue,
        "Cost": cost,
        "Profit": profit,
        "Margin %": (profit / revenue * 100) if revenue else float("nan"),
        "Units": units,
        "Orders": orders,
        "Avg Order Value": (revenue / orders) if orders else float("nan"),
    }


def pct_delta(actual: float, reference: float) -> float | None:
    if reference is None or reference == 0 or pd.isna(reference) or pd.isna(actual):
        return None
    return (actual - reference) / abs(reference) * 100


def add_period(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Add a ``Period`` timestamp column for Month / Quarter / Year roll-ups."""
    df = df.copy()
    if granularity == "Month":
        df["Period"] = df["Month"]
    elif granularity == "Quarter":
        df["Period"] = df["Date"].dt.to_period("Q").dt.to_timestamp()
    elif granularity == "Year":
        df["Period"] = pd.to_datetime(df["Year"].astype(str) + "-01-01")
    else:
        raise ValueError(f"Unknown granularity: {granularity}")
    return df


def trend(df: pd.DataFrame, granularity: str = "Month", by: str | None = None) -> pd.DataFrame:
    d = add_period(df, granularity)
    keys = ["Period"] + ([by] if by else [])
    return summarize(d, keys).sort_values(keys)


def linear_forecast(monthly: pd.DataFrame, value_col: str, periods: int = 6, window: int = 24) -> pd.DataFrame:
    """Simple linear-trend forecast on the last ``window`` months (explicitly a rough guide)."""
    s = monthly.sort_values("Period").tail(window)
    if len(s) < 6:
        return pd.DataFrame(columns=["Period", "Forecast"])
    x = np.arange(len(s))
    slope, intercept = np.polyfit(x, s[value_col].to_numpy(dtype=float), 1)
    fx = np.arange(len(s), len(s) + periods)
    dates = pd.date_range(s["Period"].iloc[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    return pd.DataFrame({"Period": dates, "Forecast": np.clip(intercept + slope * fx, 0, None)})


def budget_vs_actual(df_both: pd.DataFrame, by: str | Sequence[str], metric: str = "Revenue") -> pd.DataFrame:
    """Actual vs Budget side by side with variance and achievement %."""
    by = [by] if isinstance(by, str) else list(by)
    pivot = df_both.pivot_table(index=by, columns="Scenario", values=metric, aggfunc="sum", fill_value=0)
    for col in ("Actual", "Budget"):
        if col not in pivot.columns:
            pivot[col] = 0.0
    out = pivot[["Actual", "Budget"]].reset_index()
    out.columns.name = None
    out["Variance"] = out["Actual"] - out["Budget"]
    out["Variance %"] = _safe_div(out["Variance"], out["Budget"]) * 100
    out["Achievement %"] = _safe_div(out["Actual"], out["Budget"]) * 100
    return out


def heatmap_table(df: pd.DataFrame, rows: str, cols: str, metric: str = "Revenue") -> pd.DataFrame:
    if metric == "Orders":
        return df.pivot_table(index=rows, columns=cols, values="OrderID", aggfunc="count", fill_value=0)
    return df.pivot_table(index=rows, columns=cols, values=metric, aggfunc="sum", fill_value=0)


def rep_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per salesperson performance incl. designation, age and tenure."""
    g = summarize(df, ["SalesPerson", "Designation"])
    if "Age" in df.columns and "DOJ" in df.columns:
        extra = df.groupby("SalesPerson", as_index=False).agg(Age=("Age", "first"), DOJ=("DOJ", "first"))
        g = g.merge(extra, on="SalesPerson", how="left")
    return g.sort_values("Revenue", ascending=False)


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------
def fmt_num(x: float, prefix: str = "", decimals: int = 2) -> str:
    """Compact number formatting: 1.23K / 4.56M / 7.89B."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    sign = "-" if x < 0 else ""
    a = abs(float(x))
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{sign}{prefix}{a / div:.{decimals}f}{suf}"
    return f"{sign}{prefix}{a:,.0f}"


def fmt_pct(x: float, decimals: int = 1) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:.{decimals}f}%"


# --------------------------------------------------------------------------------------
# Automatic insights
# --------------------------------------------------------------------------------------
def generate_insights(df: pd.DataFrame, df_both: pd.DataFrame | None = None, prefix: str = "") -> list[str]:
    """Plain-language findings for the currently filtered data (one scenario)."""
    if df.empty:
        return ["No data matches the current filters."]
    out: list[str] = []
    k = kpis(df)
    out.append(f"Total revenue is **{fmt_num(k['Revenue'], prefix)}** from {k['Orders']:,} orders, "
               f"with a profit margin of **{fmt_pct(k['Margin %'])}**.")

    for dim, label in (("Category", "category"), ("City", "city"), ("Brand", "brand"), ("SalesPerson", "sales rep")):
        s = summarize(df, dim).sort_values("Revenue", ascending=False)
        if len(s) >= 2:
            top, share = s.iloc[0], s.iloc[0]["Revenue"] / s["Revenue"].sum() * 100
            out.append(f"Top {label} by revenue: **{top[dim]}** ({fmt_num(top['Revenue'], prefix)}, {share:.1f}% of total).")

    cat = summarize(df, "Category").sort_values("Margin %")
    if len(cat) >= 2:
        out.append(f"Highest-margin category is **{cat.iloc[-1]['Category']}** ({cat.iloc[-1]['Margin %']:.1f}%); "
                   f"lowest is **{cat.iloc[0]['Category']}** ({cat.iloc[0]['Margin %']:.1f}%).")

    yearly = summarize(df, "Year").sort_values("Year")
    if len(yearly) >= 2:
        best = yearly.loc[yearly["Revenue"].idxmax()]
        out.append(f"Best year: **{int(best['Year'])}** with {fmt_num(best['Revenue'], prefix)} revenue.")
        last, prev = yearly.iloc[-1], yearly.iloc[-2]
        g = pct_delta(last["Revenue"], prev["Revenue"])
        if g is not None:
            direction = "up" if g >= 0 else "down"
            out.append(f"Revenue in {int(last['Year'])} is **{direction} {abs(g):.1f}%** versus {int(prev['Year'])}.")

    m = summarize(df, "MonthName")
    if len(m) >= 2:
        out.append(f"Seasonally strongest month: **{m.loc[m['Revenue'].idxmax(), 'MonthName']}**; "
                   f"weakest: **{m.loc[m['Revenue'].idxmin(), 'MonthName']}** (summed across years).")

    if df_both is not None and {"Actual", "Budget"} <= set(df_both["Scenario"].unique()):
        bva = budget_vs_actual(df_both, "Year")
        tot_a, tot_b = bva["Actual"].sum(), bva["Budget"].sum()
        ach = tot_a / tot_b * 100 if tot_b else float("nan")
        out.append(f"Overall Actual revenue is **{fmt_pct(ach)}** of Budget "
                   f"({'above' if tot_a >= tot_b else 'below'} plan by {fmt_num(abs(tot_a - tot_b), prefix)}).")
        city = budget_vs_actual(df_both, "City")
        if len(city) >= 2:
            w, l = city.loc[city["Achievement %"].idxmax()], city.loc[city["Achievement %"].idxmin()]
            out.append(f"Budget achievement is strongest in **{w['City']}** ({w['Achievement %']:.1f}%) "
                       f"and weakest in **{l['City']}** ({l['Achievement %']:.1f}%).")
    return out
