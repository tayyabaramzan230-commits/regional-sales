"""Regional Sales – ML Analytics & Forecasting (Streamlit).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ml_utils as ml
import sales_utils as su

# --------------------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Sales ML Studio", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

PALETTE = ["#2563eb", "#0d9488", "#f59e0b", "#e11d48", "#7c3aed", "#64748b", "#0ea5e9", "#84cc16"]
SCEN_COLORS = {"Actual": "#2563eb", "Budget": "#94a3b8"}

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px;}
      div[data-testid="stMetric"] {background: rgba(148,163,184,.10); border: 1px solid rgba(148,163,184,.25);
                                   border-radius: 12px; padding: 14px 16px;}
      div[data-testid="stMetricLabel"] p {font-size: .85rem; opacity: .75;}
      h1 {font-size: 1.9rem !important; margin-bottom: .1rem;}
      .subtitle {opacity: .65; margin-bottom: 1rem;}
      .big {font-size: 2.4rem; font-weight: 700; line-height: 1.1;}
      .card {background: rgba(37,99,235,.08); border: 1px solid rgba(37,99,235,.35); border-radius: 14px; padding: 18px 22px;}
      button[data-baseweb="tab"] {font-size: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading sales data…")
def load_bundled() -> pd.DataFrame:
    return su.load_data()


@st.cache_data(show_spinner="Reading uploaded file…")
def load_uploaded(name: str, content: bytes) -> pd.DataFrame:
    bio = io.BytesIO(content)
    bio.name = name
    return su.load_data(bio)


@st.cache_resource(show_spinner="Training prediction model…")
def fit_predictor(algo: str, data_key: str, _df: pd.DataFrame):
    """Revenue and Cost models for the Sales Predictor tab (Profit = Revenue − Cost)."""
    d = _df[_df["Scenario"] == "Actual"] if (_df["Scenario"] == "Actual").any() else _df
    models = {}
    for tgt in ("Revenue", "Cost"):
        dd = ml.add_features(d).dropna(subset=ml.ALL_FEATURES + [tgt])
        if len(dd) > 40_000:
            dd = dd.sample(40_000, random_state=ml.SEED)
        m = ml.make_model(algo)
        m.fit(dd[ml.ALL_FEATURES], dd[tgt])
        models[tgt] = m
    return models


def show(fig, height: int = 380):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      title=dict(x=0, font=dict(size=15)), colorway=PALETTE, font=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)


def table(df: pd.DataFrame, money=(), pct=(), dec=(), height=None):
    fmt = {c: "{:,.0f}" for c in money if c in df.columns}
    fmt.update({c: "{:.1f}%" for c in pct if c in df.columns})
    fmt.update({c: "{:,.3f}" for c in dec if c in df.columns})
    st.dataframe(df.style.format(fmt, na_rep="–"), use_container_width=True, hide_index=True, height=height)


def money(x):
    return su.fmt_num(x, cur_prefix)


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
st.sidebar.title("🤖 Sales ML Studio")
with st.sidebar.expander("Data source", expanded=False):
    uploaded = st.file_uploader("Upload your own workbook (same columns)", type=["xlsx", "csv"],
                                help="Leave empty to use the bundled Regional Sales dataset.")
    currency = st.text_input("Currency label", value="PKR", help="Display only – the dataset has no currency field.")

try:
    data = load_uploaded(uploaded.name, uploaded.getvalue()) if uploaded is not None else load_bundled()
    data_key = uploaded.name if uploaded is not None else "bundled"
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load the data: {exc}")
    st.stop()
if data.empty:
    st.error("The dataset is empty.")
    st.stop()

cur_prefix = f"{currency} " if currency else ""
data = data.assign(Orders=1)

st.sidebar.markdown("### Filters for Analysis & Visualization")
scen_opts = ["All"] + sorted(data["Scenario"].unique(), key=lambda s: (s != "Actual", s))
scen_sel = st.sidebar.radio("Scenario", scen_opts, index=1 if len(scen_opts) > 1 else 0, horizontal=True, key="f_scen")
min_d, max_d = data["Date"].min().date(), data["Date"].max().date()
date_sel = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d, key="f_date")
if isinstance(date_sel, (tuple, list)):
    d0, d1 = date_sel[0], (date_sel[1] if len(date_sel) > 1 else date_sel[0])
else:
    d0 = d1 = date_sel


def multi(label, col):
    opts = sorted(data[col].unique())
    return st.sidebar.multiselect(label, opts, default=opts, key=f"f_{col}")


f_city, f_cat, f_brand, f_rep = multi("City", "City"), multi("Category", "Category"), multi("Brand", "Brand"), multi("Sales person", "SalesPerson")
st.sidebar.caption("Training, the Sales Predictor and Forecasting have their own controls and use all history.")

mask = (
    (data["Date"] >= pd.Timestamp(d0)) & (data["Date"] <= pd.Timestamp(d1) + pd.Timedelta(hours=23, minutes=59))
    & data["City"].isin(f_city) & data["Category"].isin(f_cat) & data["Brand"].isin(f_brand) & data["SalesPerson"].isin(f_rep)
)
if scen_sel != "All":
    mask &= data["Scenario"] == scen_sel
df = data[mask]

# --------------------------------------------------------------------------------------
# Header & tabs
# --------------------------------------------------------------------------------------
st.title("Sales ML Studio")
st.markdown("<div class='subtitle'>Explore · visualise · predict · forecast — Regional Sales data "
            f"({data['Date'].min():%b %Y} – {data['Date'].max():%b %Y})</div>", unsafe_allow_html=True)

t_an, t_viz, t_train, t_pred, t_fc, t_about = st.tabs(
    ["📋 Data analysis", "📊 Visualization", "🧠 Predictive modelling", "🎯 Sales predictor", "🔮 Forecasting", "ℹ️ About"]
)

# ======================================================================================
# 1. DATA ANALYSIS
# ======================================================================================
with t_an:
    raw = data.drop(columns=["Orders"])
    c = st.columns(5)
    c[0].metric("Rows", f"{len(raw):,}")
    c[1].metric("Columns", f"{raw.shape[1]}")
    c[2].metric("Missing values", f"{int(raw.isna().sum().sum()):,}")
    c[3].metric("Duplicate rows", f"{int(raw.duplicated().sum()):,}")
    c[4].metric("Date span", f"{(data['Date'].max() - data['Date'].min()).days // 365 + 1} yrs")

    if df.empty:
        st.warning("No rows match the sidebar filters – the sections below are empty.")
    else:
        k = su.kpis(df)
        c = st.columns(6)
        c[0].metric("Revenue", money(k["Revenue"]))
        c[1].metric("Profit", money(k["Profit"]))
        c[2].metric("Margin", su.fmt_pct(k["Margin %"]))
        c[3].metric("Units", f"{k['Units']:,}")
        c[4].metric("Orders", f"{k['Orders']:,}")
        c[5].metric("Avg order value", money(k["Avg Order Value"]))

        with st.expander("💡 Auto-generated insights", expanded=True):
            one = df[df["Scenario"] == (scen_sel if scen_sel != "All" else "Actual")]
            if not one.empty:
                both = data[mask & data["Scenario"].isin(["Actual", "Budget"])] if scen_sel == "All" else None
                for line in su.generate_insights(one, both, cur_prefix):
                    st.markdown(f"- {line}")

        a, b = st.columns(2)
        with a:
            st.markdown("**Column profile**")
            prof = pd.DataFrame({"Column": raw.columns, "Type": [str(t) for t in raw.dtypes],
                                 "Unique": [raw[c_].nunique() for c_ in raw.columns],
                                 "Missing": [int(raw[c_].isna().sum()) for c_ in raw.columns],
                                 "Example": [str(raw[c_].iloc[0]) for c_ in raw.columns]})
            st.dataframe(prof, use_container_width=True, hide_index=True, height=360)
        with b:
            st.markdown("**Descriptive statistics (filtered)**")
            num = [c_ for c_ in ["Units", "UnitPrice", "UnitCost", "Revenue", "Cost", "Profit"] if c_ in df.columns]
            st.dataframe(df[num].describe().T.round(1).reset_index().rename(columns={"index": "Metric"}),
                         use_container_width=True, hide_index=True, height=360)

        a, b = st.columns(2)
        with a:
            corr_cols = [c_ for c_ in ["Units", "UnitPrice", "UnitCost", "Revenue", "Cost", "Profit", "Year", "MonthNum", "Age"]
                         if c_ in df.columns]
            corr = df[corr_cols].corr().round(2)
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            title="Correlation matrix")
            show(fig, 430)
        with b:
            dist_col = st.selectbox("Distribution of", ["Revenue", "Profit", "Units", "UnitPrice"], key="an_dist")
            fig = px.histogram(df, x=dist_col, nbins=60, color_discrete_sequence=[PALETTE[0]],
                               title=f"Distribution of {dist_col}")
            fig.update_xaxes(title=None)
            show(fig, 360)
            q1, q3 = df[dist_col].quantile([0.25, 0.75])
            iqr = q3 - q1
            n_out = int(((df[dist_col] < q1 - 1.5 * iqr) | (df[dist_col] > q3 + 1.5 * iqr)).sum())
            st.caption(f"{n_out:,} high/low outliers by the 1.5×IQR rule ({n_out / len(df) * 100:.1f}% of rows). "
                       f"Skewness: {df[dist_col].skew():.2f}.")

        st.markdown("**Group analysis**")
        g1, g2 = st.columns([1, 3])
        gdim = g1.selectbox("Group by", ["Category", "Brand", "Product", "City", "SalesPerson", "Designation", "Warranty",
                                         "Year", "Quarter", "MonthName", "Scenario"], key="an_group")
        gt = su.summarize(df, gdim).sort_values("Revenue", ascending=False)
        with g2:
            table(gt, money=["Revenue", "Cost", "Profit", "Units", "Orders", "Avg Order Value"], pct=["Margin %"], height=300)

        st.markdown("**Anomalous months** (seasonally-adjusted robust z-score)")
        one = df[df["Scenario"] == (scen_sel if scen_sel != "All" else "Actual")]
        s_all = ml.monthly_series(one, "Revenue")
        if len(s_all) >= 12:
            an = ml.detect_anomalies(s_all)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(an.index), y=an["Value"], mode="lines", name="Monthly revenue",
                                     line=dict(color=PALETTE[0])))
            flagged = an[an["Anomaly"]]
            fig.add_trace(go.Scatter(x=list(flagged.index), y=flagged["Value"], mode="markers", name="Anomaly",
                                     marker=dict(color=PALETTE[3], size=11, symbol="diamond")))
            fig.update_layout(title="Monthly revenue with flagged anomalies")
            fig.update_yaxes(tickformat="~s")
            show(fig, 330)
            if flagged.empty:
                st.caption("No unusual months detected.")

# ======================================================================================
# 2. VISUALIZATION
# ======================================================================================
with t_viz:
    if df.empty:
        st.warning("No rows match the sidebar filters.")
    else:
        l, m_, r_ = st.columns([2, 2, 3])
        vm = l.radio("Metric", ["Revenue", "Profit", "Units", "Orders"], horizontal=True, key="vz_metric")
        gran = m_.radio("Granularity", ["Month", "Quarter", "Year"], horizontal=True, key="vz_gran")
        split_by = r_.selectbox("Split trend by", ["(none)", "Scenario", "Category", "City", "Designation", "Warranty"], key="vz_split")

        tr = su.trend(df, gran, by=None if split_by == "(none)" else split_by)
        fig = px.line(tr, x="Period", y=vm, color=None if split_by == "(none)" else split_by,
                      color_discrete_sequence=PALETTE, title=f"{vm} by {gran.lower()}")
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None, tickformat="~s")
        show(fig, 400)

        a, b = st.columns(2)
        with a:
            cat = su.summarize(df, "Category").sort_values(vm)
            fig = px.bar(cat, x=vm, y="Category", orientation="h", color_discrete_sequence=[PALETTE[0]],
                         text_auto=".3s", title=f"{vm} by category")
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            show(fig)
        with b:
            cty = su.summarize(df, "City").sort_values(vm)
            fig = px.bar(cty, x=vm, y="City", orientation="h", color_discrete_sequence=[PALETTE[1]],
                         text_auto=".3s", title=f"{vm} by city")
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            show(fig)

        a, b = st.columns(2)
        with a:
            tm = su.summarize(df, ["Category", "Brand"])
            tm = tm[tm["Revenue"] > 0]
            fig = px.treemap(tm, path=["Category", "Brand"], values="Revenue", color="Margin %",
                             color_continuous_scale="Blues", title="Revenue mix (colour = margin %)")
            show(fig, 400)
        with b:
            hm = su.heatmap_table(df, "City", "Category", vm)
            fig = px.imshow(hm, text_auto=".3s", aspect="auto", color_continuous_scale="Blues",
                            title=f"{vm}: city × category")
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            show(fig, 400)

        a, b = st.columns(2)
        with a:
            hm2 = su.heatmap_table(df, "Year", "MonthName", vm).reindex(columns=[m for m in su.MONTH_NAMES])
            hm2.index = hm2.index.astype(str)
            fig = px.imshow(hm2, text_auto=".2s", aspect="auto", color_continuous_scale="Blues",
                            title=f"{vm}: year × month (seasonality)")
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            show(fig, 380)
        with b:
            sample = df.sample(min(4000, len(df)), random_state=1)
            fig = px.box(sample, x="Category", y="Revenue", color="Category", color_discrete_sequence=PALETTE,
                         title="Order revenue distribution by category (sample)")
            fig.update_layout(showlegend=False)
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None, tickformat="~s")
            show(fig, 380)

        a, b = st.columns(2)
        with a:
            ps = su.summarize(df, ["Product", "Category"])
            ps = ps[(ps["Revenue"] > 0) & (ps["Units"] > 0)]
            fig = px.scatter(ps, x="Revenue", y="Margin %", size="Units", color="Category", hover_name="Product",
                             size_max=36, color_discrete_sequence=PALETTE, title="Products: revenue vs margin (bubble = units)")
            fig.update_xaxes(tickformat="~s")
            show(fig, 400)
        with b:
            pr = df.groupby(["Product", "Year"], as_index=False)["UnitPrice"].median()
            top_products = su.summarize(df, "Product").sort_values("Revenue", ascending=False)["Product"].head(6)
            pr = pr[pr["Product"].isin(top_products)]
            pr["Year"] = pr["Year"].astype(str)
            fig = px.line(pr, x="Year", y="UnitPrice", color="Product", markers=True, color_discrete_sequence=PALETTE,
                          title="Median unit price over time (top 6 products)")
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None, tickformat="~s")
            show(fig, 400)

        rp = su.rep_profile(df).sort_values(vm if vm in ("Revenue", "Profit", "Units", "Orders") else "Revenue")
        fig = px.bar(rp, x=vm, y="SalesPerson", color="Designation", orientation="h", color_discrete_sequence=PALETTE,
                     text_auto=".3s", title=f"Sales team – {vm}")
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None)
        show(fig, 120 + 30 * len(rp))

# ======================================================================================
# 3. PREDICTIVE MODELLING
# ======================================================================================
with t_train:
    st.markdown("### Train & compare models that predict the sales value of an order")
    st.caption("Features: product, category, brand, warranty, city, sales person, designation, scenario, units and calendar fields. "
               "The best model can then be used in the **Sales predictor** tab.")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    target = c1.selectbox("Predict", ml.TARGETS, key="tr_target")
    split = c2.radio("Validation split", ["time", "random"], horizontal=True, key="tr_split",
                     help="time = train on the past, test on the most recent orders (realistic). "
                          "random = shuffled rows (can leak information).")
    test_size = c3.slider("Test share", 0.1, 0.4, 0.2, 0.05, key="tr_test")
    model_sel = c4.multiselect("Models", ml.MODEL_NAMES, default=["Unit-Price Trend (hybrid)", "Gradient Boosting", "Ridge (linear)"],
                               key="tr_models")
    train_scen = st.radio("Train on scenario", ["Actual", "Budget", "Both"], horizontal=True, key="tr_scen")
    if split == "random":
        st.warning("A random split lets tree models memorise the yearly price list, so scores look near-perfect (R² ≈ 1.0) "
                   "but will not hold for future dates. Prefer the **time** split.")

    if st.button("🚀 Train models", type="primary", key="tr_btn"):
        if not model_sel:
            st.error("Select at least one model.")
        else:
            tdf = data if train_scen == "Both" else data[data["Scenario"] == train_scen]
            bar = st.progress(0.0, text="Training…")
            try:
                res = ml.train_models(tdf, target, model_sel, split, test_size, progress=lambda f, n: bar.progress(f, text=f"Trained {n}"))
                st.session_state["train"] = (res, tdf)
                st.session_state.pop("importance", None)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Training failed: {exc}")
            bar.empty()

    if "train" in st.session_state:
        res, tdf = st.session_state["train"]
        st.success(f"Best model on the test set: **{res.best_model}** · target = {res.target} · "
                   f"{res.n_train:,} train / {res.n_test:,} test rows ({res.split} split)")
        best_row = res.metrics.iloc[0]
        c = st.columns(4)
        c[0].metric("R² (best)", f"{best_row['R²']:.4f}")
        c[1].metric("MAE", su.fmt_num(best_row["MAE"], cur_prefix))
        c[2].metric("RMSE", su.fmt_num(best_row["RMSE"], cur_prefix))
        c[3].metric("MAPE", f"{best_row['MAPE %']:.2f}%")
        table(res.metrics, pct=[], dec=["R²"], money=["MAE", "RMSE"])

        mcol = st.selectbox("Inspect model", list(res.pipelines), index=list(res.pipelines).index(res.best_model), key="tr_inspect")
        pc = f"Pred – {mcol}"
        pred = res.predictions
        a, b = st.columns(2)
        with a:
            samp = pred.sample(min(2500, len(pred)), random_state=2)
            lim = float(max(samp[res.target].max(), samp[pc].max()))
            fig = px.scatter(samp, x=res.target, y=pc, opacity=0.45, color_discrete_sequence=[PALETTE[0]],
                             hover_data={"Product": True}, title=f"Actual vs predicted {res.target.lower()} (test sample)")
            fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="Perfect fit", line=dict(color=PALETTE[3], dash="dash")))
            fig.update_xaxes(title="Actual", tickformat="~s")
            fig.update_yaxes(title="Predicted", tickformat="~s")
            show(fig, 400)
        with b:
            err = pred.assign(Error=pred[pc] - pred[res.target])
            fig = px.histogram(err, x="Error", nbins=60, color_discrete_sequence=[PALETTE[4]], title="Prediction error distribution")
            show(fig, 400)

        a, b = st.columns(2)
        with a:
            mt = pred.assign(Month=pred["Date"].dt.to_period("M").dt.to_timestamp()).groupby("Month", as_index=False)[[res.target, pc]].sum()
            long = mt.melt(id_vars="Month", value_vars=[res.target, pc], var_name="Series", value_name="Value")
            long["Series"] = long["Series"].map({res.target: "Actual", pc: "Predicted"})
            fig = px.line(long, x="Month", y="Value", color="Series", color_discrete_map={"Actual": PALETTE[0], "Predicted": PALETTE[2]},
                          title="Test period: monthly total, actual vs predicted")
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None, tickformat="~s")
            show(fig, 360)
        with b:
            ec = pred.assign(AbsPct=(pred[pc] - pred[res.target]).abs() / pred[res.target].replace(0, np.nan) * 100)
            ec = ec.groupby("Category", as_index=False)["AbsPct"].mean().sort_values("AbsPct")
            fig = px.bar(ec, x="AbsPct", y="Category", orientation="h", color_discrete_sequence=[PALETTE[3]],
                         text_auto=".1f", title="Mean absolute % error by category")
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            show(fig, 360)

        st.markdown("**Feature importance** (permutation importance on the test period)")
        if st.button("Compute importance", key="tr_imp_btn"):
            with st.spinner("Computing…"):
                st.session_state["importance"] = (mcol, ml.feature_importance(res, mcol, tdf))
        if "importance" in st.session_state and st.session_state["importance"][0] == mcol:
            imp = st.session_state["importance"][1].copy()
            tot = imp["Importance"].clip(lower=0).sum() or 1
            imp["Share %"] = imp["Importance"].clip(lower=0) / tot * 100
            fig = px.bar(imp.sort_values("Share %"), x="Share %", y="Feature", orientation="h", color_discrete_sequence=[PALETTE[1]],
                         text_auto=".1f", title=f"What drives {res.target.lower()} predictions? ({mcol})")
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            show(fig, 380)

        d1_, d2_ = st.columns(2)
        d1_.download_button("⬇️ Test-set predictions (CSV)", pred.to_csv(index=False).encode("utf-8"),
                            file_name="test_predictions.csv", mime="text/csv", use_container_width=True)
        d2_.download_button("⬇️ Trained model (.joblib)", ml.model_to_bytes(res.pipelines[mcol], {"target": res.target, "model": mcol}),
                            file_name=f"model_{res.target.lower()}.joblib", mime="application/octet-stream", use_container_width=True,
                            help="Load with joblib.load(); keep ml_utils.py importable.")
        st.info("**Why the hybrid model wins:** list prices step up each year, and tree models can only predict price levels they "
                "have seen. The hybrid multiplies units by a per-product price *trend*, so it extrapolates to later dates.")

# ======================================================================================
# 4. SALES PREDICTOR
# ======================================================================================
with t_pred:
    st.markdown("### Predict the sales value of a future order")
    algo = st.selectbox("Prediction model", ["Unit-Price Trend (hybrid)", "Gradient Boosting", "Random Forest"], key="sp_algo",
                        help="The hybrid model is instant and extrapolates price trends. Others train once (≈5–15 s) and are cached.")
    models = fit_predictor(algo, data_key, data)
    lookup = ml.product_lookup(data)
    rep_map = data.groupby("SalesPerson")["Designation"].first()

    left, right = st.columns([2, 3])
    with left:
        product = st.selectbox("Product", list(lookup.index), key="sp_prod")
        info = lookup.loc[product]
        st.caption(f"{info['Category']} · {info['Brand']} · {info['Warranty']} warranty")
        city = st.selectbox("City", sorted(data["City"].unique()), key="sp_city")
        rep = st.selectbox("Sales person", list(rep_map.index), key="sp_rep")
        units = st.number_input("Units", min_value=1, max_value=100, value=3, step=1, key="sp_units")
        default_date = (data["Date"].max() + pd.Timedelta(days=15)).date()
        when = st.date_input("Order date", value=default_date, min_value=data["Date"].min().date(),
                             max_value=pd.Timestamp("2035-12-31").date(), key="sp_date")
        scen_p = st.radio("Scenario", ["Actual", "Budget"], horizontal=True, key="sp_scen")

    row = ml.build_order_row(lookup, product, city, rep, rep_map[rep], units, when, scen_p)
    p_rev = float(ml.predict_orders(models["Revenue"], row)[0])
    p_cost = float(ml.predict_orders(models["Cost"], row)[0])
    p_profit = p_rev - p_cost

    with right:
        st.markdown(f"<div class='card'><div style='opacity:.7'>Predicted revenue</div><div class='big'>{money(p_rev)}</div>"
                    f"<div style='opacity:.7;margin-top:6px'>for {units} × {product} · {city} · {pd.Timestamp(when):%d %b %Y}</div></div>",
                    unsafe_allow_html=True)
        c = st.columns(3)
        c[0].metric("Predicted cost", money(p_cost))
        c[1].metric("Predicted profit", money(p_profit))
        c[2].metric("Margin", su.fmt_pct(p_profit / p_rev * 100 if p_rev else float("nan")))
        hist_p = data[(data["Product"] == product) & (data["Scenario"] == scen_p)]
        if not hist_p.empty:
            last_year = hist_p["Year"].max()
            ref = hist_p[hist_p["Year"] == last_year]["UnitPrice"].median()
            st.caption(f"Reference: median unit price for this product in {last_year} was {money(ref)}; "
                       f"the model implies {money(p_rev / units)} per unit on this date.")

    a, b = st.columns(2)
    with a:
        grid_dates = pd.date_range("2015-07-01", "2024-07-01", freq="12MS")
        rows = pd.concat([ml.build_order_row(lookup, product, city, rep, rep_map[rep], 1, d, scen_p) for d in grid_dates])
        curve = pd.DataFrame({"Year": grid_dates.year, "Predicted price per unit": ml.predict_orders(models["Revenue"], rows)})
        hist_curve = (data[(data["Product"] == product) & (data["Scenario"] == scen_p)].groupby("Year", as_index=False)["UnitPrice"].median())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["Year"], y=curve["Predicted price per unit"], mode="lines+markers", name="Model (incl. future)",
                                 line=dict(color=PALETTE[0], dash="dot")))
        fig.add_trace(go.Scatter(x=hist_curve["Year"], y=hist_curve["UnitPrice"], mode="markers", name="Historical median",
                                 marker=dict(color=PALETTE[2], size=10)))
        fig.update_layout(title="Revenue per unit over time")
        fig.update_yaxes(tickformat="~s")
        show(fig, 360)
    with b:
        u = np.arange(1, 11)
        rows = pd.concat([ml.build_order_row(lookup, product, city, rep, rep_map[rep], x, when, scen_p) for x in u])
        w = pd.DataFrame({"Units": u, "Revenue": ml.predict_orders(models["Revenue"], rows),
                          "Cost": ml.predict_orders(models["Cost"], rows)})
        w["Profit"] = w["Revenue"] - w["Cost"]
        fig = px.line(w.melt(id_vars="Units", var_name="Measure", value_name="Value"), x="Units", y="Value", color="Measure",
                      markers=True, color_discrete_sequence=PALETTE, title="What-if: order size")
        fig.update_yaxes(tickformat="~s")
        show(fig, 360)

    st.markdown("**Compare cities** for the same order")
    rows = pd.concat([ml.build_order_row(lookup, product, c_, rep, rep_map[rep], units, when, scen_p) for c_ in sorted(data["City"].unique())])
    cc = pd.DataFrame({"City": sorted(data["City"].unique()), "Predicted revenue": ml.predict_orders(models["Revenue"], rows)})
    table(cc, money=["Predicted revenue"])

    st.markdown("### Batch predictions")
    tmpl = pd.DataFrame({"Date": [str(default_date)] * 2, "Product": [list(lookup.index)[0], list(lookup.index)[1]],
                         "City": [sorted(data["City"].unique())[0]] * 2, "SalesPerson": [list(rep_map.index)[0]] * 2,
                         "Units": [2, 5], "Scenario": ["Actual", "Actual"]})
    st.download_button("⬇️ Download CSV template", tmpl.to_csv(index=False).encode("utf-8"), file_name="batch_template.csv", mime="text/csv")
    up = st.file_uploader("Upload a CSV with columns Date, Product, City, SalesPerson, Units (Scenario optional)", type=["csv"], key="sp_batch")
    if up is not None:
        try:
            b_df = pd.read_csv(up)
            need = {"Date", "Product", "City", "SalesPerson", "Units"}
            if not need <= set(b_df.columns):
                st.error(f"Missing columns: {sorted(need - set(b_df.columns))}")
            else:
                b_df["Date"] = pd.to_datetime(b_df["Date"])
                bad = b_df[~b_df["Product"].isin(lookup.index) | ~b_df["SalesPerson"].isin(rep_map.index)]
                if len(bad):
                    st.warning(f"{len(bad)} row(s) have an unknown Product or SalesPerson and were skipped.")
                b_df = b_df.drop(index=bad.index)
                if "Scenario" not in b_df.columns:
                    b_df["Scenario"] = "Actual"
                full = pd.concat([ml.build_order_row(lookup, r["Product"], r["City"], r["SalesPerson"], rep_map[r["SalesPerson"]],
                                                     r["Units"], r["Date"], r["Scenario"]) for _, r in b_df.iterrows()], ignore_index=True)
                out = b_df.reset_index(drop=True).copy()
                out["Predicted Revenue"] = ml.predict_orders(models["Revenue"], full).round(0)
                out["Predicted Cost"] = ml.predict_orders(models["Cost"], full).round(0)
                out["Predicted Profit"] = out["Predicted Revenue"] - out["Predicted Cost"]
                st.dataframe(out, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode("utf-8"), file_name="batch_predictions.csv",
                                   mime="text/csv")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not process the file: {exc}")

# ======================================================================================
# 6. ABOUT
# ======================================================================================
with t_about:
    st.markdown(
        """
### How it works

**Data** – 100,000 order lines (Actual vs Budget scenarios, 2015-2020). The loader auto-detects the header row, fixes types,
and derives *Profit = Revenue − Cost* plus calendar fields.

**Predictive modelling** – regression models predict the revenue / cost / profit of a single order from product, location,
sales person, units and date. Models are validated on a **time-ordered hold-out** (train on the past, test on the latest orders).
* *Unit-Price Trend (hybrid)*: units × a per-product linear price trend. Extrapolates to future dates.
* *Gradient Boosting / Random Forest / Extra Trees*: non-linear tree ensembles (cannot extrapolate beyond price levels seen in training).
* *Ridge*: linear baseline (cannot represent the units × price interaction – shown for comparison).

**Forecasting** – monthly series are forecast with four methods (seasonal naive, trend + seasonality regression, additive Holt-Winters,
and gradient boosting on lag features) plus their average. Each is back-tested on the last N months; *Auto* uses the one with the lowest RMSE,
then refits on all history. Prediction intervals come from the backtest errors.

### Caveats
* In this dataset each product's unit price depends only on the **year**, so order-level revenue is highly predictable;
  expect lower accuracy on noisier real-world data.
* Forecasts assume past patterns continue (trend and seasonality). They do not know about promotions, new products or shocks.
* A random validation split overstates accuracy – use the time split.
* The currency label is display-only.
        """
    )

# ======================================================================================
# 5. FORECASTING
# ======================================================================================
with t_fc:
    st.markdown("### Forecast future sales")
    c1, c2, c3, c4 = st.columns(4)
    fmetric = c1.selectbox("Metric", ["Revenue", "Profit", "Units", "Orders"], key="fc_metric")
    horizon = c2.slider("Horizon (months)", 3, 24, 12, key="fc_h")
    seg_dim = c3.selectbox("Segment", ["Total company", "City", "Category", "Brand", "SalesPerson", "Designation", "Warranty", "Product"], key="fc_dim")
    seg_val = None
    if seg_dim != "Total company":
        seg_val = c4.selectbox(seg_dim, sorted(data[seg_dim].unique()), key="fc_val")
    d1, d2 = st.columns([2, 1])
    method_choice = d1.selectbox("Method", ["Auto (best on backtest)"] + list(ml.METHODS) + [ml.ENSEMBLE], key="fc_method")
    holdout = d2.slider("Backtest months", 6, 18, 12, key="fc_hold")

    base = data[data["Scenario"] == "Actual"] if (data["Scenario"] == "Actual").any() else data
    seg = base if seg_val is None else base[base[seg_dim] == seg_val]
    series = ml.monthly_series(seg, fmetric)

    try:
        fr = ml.forecast(series, horizon, "auto" if method_choice.startswith("Auto") else method_choice, holdout)
    except ValueError as exc:
        st.warning(f"{exc} Pick a broader segment or a shorter backtest window.")
        st.stop()  # (About tab is rendered above, so only the rest of this tab is skipped)

    fdf = fr.forecast
    recent = float(series.iloc[-horizon:].sum())
    nxt = float(fdf["Forecast"].sum())
    is_money = fmetric in ("Revenue", "Profit")
    fmt = (lambda v: money(v)) if is_money else (lambda v: f"{v:,.0f}")
    c = st.columns(4)
    c[0].metric(f"Next {horizon} months (forecast)", fmt(nxt), f"{(nxt - recent) / recent * 100:+.1f}% vs last {horizon}m" if recent else None)
    c[1].metric(f"Last {horizon} months (actual)", fmt(recent))
    c[2].metric("Method", fr.method.split(" (")[0].replace("ML: ", ""))
    best_row = fr.scores[fr.scores["Method"] == fr.method].iloc[0]
    c[3].metric("Backtest MAPE", f"{best_row['MAPE %']:.1f}%")

    fig = go.Figure()
    idx = list(fdf.index)
    for lvl, col in ((95, "rgba(37,99,235,0.10)"), (80, "rgba(37,99,235,0.20)")):
        fig.add_trace(go.Scatter(x=idx + idx[::-1], y=list(fdf[f"Upper{lvl}"]) + list(fdf[f"Lower{lvl}"])[::-1], fill="toself",
                                 fillcolor=col, line=dict(width=0), hoverinfo="skip", name=f"{lvl}% interval"))
    fig.add_trace(go.Scatter(x=list(series.index), y=list(series.values), mode="lines", name="History (Actual)",
                             line=dict(color=PALETTE[5], width=2)))
    if (data["Scenario"] == "Budget").any():
        bser = ml.monthly_series(data[(data["Scenario"] == "Budget") & ((data[seg_dim] == seg_val) if seg_val is not None else True)], fmetric)
        fig.add_trace(go.Scatter(x=list(bser.index), y=list(bser.values), mode="lines", name="Budget (reference)",
                                 line=dict(color="#cbd5e1", width=1.5, dash="dot"), visible="legendonly"))
    fig.add_trace(go.Scatter(x=idx, y=list(fdf["Forecast"]), mode="lines+markers", name="Forecast",
                             line=dict(color=PALETTE[0], width=3)))
    fig.update_layout(title=f"{fmetric} forecast – {seg_val or 'Total company'} ({fr.method})")
    fig.update_yaxes(tickformat="~s")
    show(fig, 440)
    st.caption("Intervals are approximate: they are derived from the errors of the backtest and widen slightly with the horizon.")

    a, b = st.columns(2)
    with a:
        st.markdown(f"**Backtest – last {holdout} months held out**")
        table(fr.scores, pct=["MAPE %", "Bias %"], money=["MAE", "RMSE"])
        st.caption("Lower RMSE/MAPE is better. Bias % > 0 means the method over-forecasts.")
    with b:
        bt = fr.backtest_fc.reset_index().rename(columns={"index": "Month"})
        long = bt.melt(id_vars="Month", var_name="Series", value_name="Value")
        fig = px.line(long, x="Month", y="Value", color="Series", color_discrete_sequence=["#0f172a"] + PALETTE,
                      title="Backtest: methods vs actual")
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None, tickformat="~s")
        show(fig, 330)

    st.markdown("**Forecast table**")
    ft = fdf.reset_index().rename(columns={"index": "Month"})
    ft["Month"] = ft["Month"].dt.strftime("%b %Y")
    table(ft, money=["Forecast", "Lower80", "Upper80", "Lower95", "Upper95"])
    st.download_button("⬇️ Download forecast (CSV)", ft.to_csv(index=False).encode("utf-8"), file_name="forecast.csv", mime="text/csv")

    st.markdown("### Forecast every segment at once")
    sg1, sg2 = st.columns([1, 3])
    sdim = sg1.selectbox("Segment by", ["City", "Category", "Brand", "SalesPerson", "Product", "Designation"], key="fc_all_dim")
    st_tbl = ml.segment_forecast_table(base, sdim, fmetric, horizon)
    with sg2:
        if st_tbl.empty:
            st.info("Not enough history per segment.")
        else:
            col_next = f"Next {horizon}m (forecast)"
            fig = px.bar(st_tbl.sort_values(col_next), x=col_next, y=sdim, orientation="h", color="Growth %",
                         color_continuous_scale="RdYlGn", color_continuous_midpoint=0, title=f"Forecast {fmetric.lower()} by {sdim.lower()} (colour = growth vs last {horizon}m)")
            fig.update_xaxes(title=None, tickformat="~s")
            fig.update_yaxes(title=None)
            show(fig, 120 + 30 * len(st_tbl))
    if not st_tbl.empty:
        table(st_tbl, money=[f"Last {horizon}m", f"Next {horizon}m (forecast)"], pct=["Growth %"])

