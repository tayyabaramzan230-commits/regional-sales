"""Machine-learning helpers: order-level sales prediction and time-series forecasting.

Pure pandas / numpy / scikit-learn / scipy (no Streamlit), so everything is unit-testable.
"""
from __future__ import annotations

import io
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42

# ======================================================================================
# PART 1 – ORDER-LEVEL PREDICTIVE MODELS
# ======================================================================================
CAT_FEATURES = ["Product", "Category", "Brand", "Warranty", "City", "SalesPerson", "Designation", "Scenario"]
NUM_FEATURES = ["Units", "Year", "MonthNum", "DayOfWeek", "DayOfYear"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
TARGETS = ["Revenue", "Profit", "Cost"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df


def _onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class UnitPriceTrendModel(BaseEstimator, RegressorMixin):
    """Hybrid model:  target = Units × per-unit value, where the per-unit value of every product follows a
    linear trend over the years.  Unlike tree models it *extrapolates* to future (unseen) price levels,
    which matters because list prices in this dataset step up every year."""

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        per_unit = np.asarray(y, float) / X["Units"].to_numpy(float)
        d = pd.DataFrame({"Product": X["Product"], "t": X["Year"].astype(float) + (X["MonthNum"] - 1) / 12, "v": per_unit})
        self.coef_ = {}
        for p, g in d.groupby("Product"):
            if g["t"].nunique() >= 2:
                b, a = np.polyfit(g["t"], g["v"], 1)
            else:
                a, b = g["v"].mean(), 0.0
            self.coef_[p] = (a, b)
        self.fallback_ = (float(d["v"].mean()), 0.0)
        return self

    def predict(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        t = X["Year"].astype(float) + (X["MonthNum"] - 1) / 12
        out = np.empty(len(X))
        for i, (p, tt, u) in enumerate(zip(X["Product"], t, X["Units"])):
            a, b = self.coef_.get(p, self.fallback_)
            out[i] = max(a + b * tt, 0.0) * u
        return out


def make_model(name: str, scale: bool = False):
    if name == "Unit-Price Trend (hybrid)":
        return UnitPriceTrendModel()
    pre = ColumnTransformer(
        [("cat", _onehot(), CAT_FEATURES),
         ("num", StandardScaler() if scale else "passthrough", NUM_FEATURES)]
    )
    if name == "Ridge (linear)":
        est, pre = Ridge(alpha=1.0), ColumnTransformer(
            [("cat", _onehot(), CAT_FEATURES), ("num", StandardScaler(), NUM_FEATURES)])
    elif name == "Random Forest":
        est = RandomForestRegressor(n_estimators=120, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    elif name == "Extra Trees":
        est = ExtraTreesRegressor(n_estimators=150, max_depth=22, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    elif name == "Gradient Boosting":
        est = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=SEED)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("pre", pre), ("model", est)])


MODEL_NAMES = ["Unit-Price Trend (hybrid)", "Gradient Boosting", "Random Forest", "Extra Trees", "Ridge (linear)"]


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    nz = y_true != 0
    return {
        "R²": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE %": float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.any() else float("nan"),
    }


@dataclass
class TrainResult:
    target: str
    split: str
    n_train: int
    n_test: int
    metrics: pd.DataFrame
    pipelines: dict = field(default_factory=dict)
    predictions: pd.DataFrame | None = None     # test rows + prediction columns per model
    best_model: str = ""


def train_models(df: pd.DataFrame, target: str, model_names: list[str], split: str = "time",
                 test_size: float = 0.2, max_train_rows: int = 40_000, progress=None) -> TrainResult:
    """Train & compare models.  split='time' keeps the latest ``test_size`` share of dates for testing."""
    if target not in TARGETS:
        raise ValueError(target)
    d = add_features(df).dropna(subset=ALL_FEATURES + [target])
    if len(d) < 200:
        raise ValueError("Not enough rows to train (need at least 200).")
    if split == "time":
        d = d.sort_values("Date")
        cut = int(len(d) * (1 - test_size))
        train, test = d.iloc[:cut], d.iloc[cut:]
    else:
        test = d.sample(frac=test_size, random_state=SEED)
        train = d.drop(test.index)
    if len(train) > max_train_rows:
        train = train.sample(max_train_rows, random_state=SEED)

    rows, pipes = [], {}
    preds = test[["Date", "Product", "City", "Category", "Units", target]].copy()
    for i, name in enumerate(model_names):
        t0 = time.time()
        pipe = make_model(name)
        pipe.fit(train[ALL_FEATURES], train[target])
        p = pipe.predict(test[ALL_FEATURES])
        m = regression_metrics(test[target], p)
        m.update({"Model": name, "Train time (s)": round(time.time() - t0, 1)})
        rows.append(m)
        pipes[name] = pipe
        preds[f"Pred – {name}"] = p
        if progress:
            progress((i + 1) / len(model_names), name)
    metrics = pd.DataFrame(rows)[["Model", "R²", "MAE", "RMSE", "MAPE %", "Train time (s)"]].sort_values("RMSE")
    return TrainResult(target, split, len(train), len(test), metrics.reset_index(drop=True), pipes, preds,
                       best_model=metrics.iloc[0]["Model"])


def feature_importance(res: TrainResult, model_name: str, df: pd.DataFrame, n: int = 3000, repeats: int = 3) -> pd.DataFrame:
    """Permutation importance on a sample of the test period (works for every model type)."""
    d = add_features(df).dropna(subset=ALL_FEATURES + [res.target]).sort_values("Date")
    test = d.iloc[-res.n_test:]
    test = test.sample(min(n, len(test)), random_state=SEED)
    r = permutation_importance(res.pipelines[model_name], test[ALL_FEATURES], test[res.target],
                               n_repeats=repeats, random_state=SEED, scoring="r2", n_jobs=1)
    out = pd.DataFrame({"Feature": ALL_FEATURES, "Importance": r.importances_mean, "Std": r.importances_std})
    return out.sort_values("Importance", ascending=False).reset_index(drop=True)


def product_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """Product → Category / Brand / Warranty plus reference price/cost statistics."""
    g = df.groupby("Product").agg(Category=("Category", "first"), Brand=("Brand", "first"),
                                  Warranty=("Warranty", "first"), MedianPrice=("UnitPrice", "median"),
                                  MedianUnitCost=("UnitCost", "median")).reset_index()
    return g.set_index("Product")


def predict_orders(pipe, rows: pd.DataFrame) -> np.ndarray:
    return np.clip(pipe.predict(add_features(rows)[ALL_FEATURES]), 0, None)


def build_order_row(lookup: pd.DataFrame, product: str, city: str, rep: str, designation: str, units: int,
                    date, scenario: str = "Actual") -> pd.DataFrame:
    info = lookup.loc[product]
    return pd.DataFrame([{
        "Date": pd.Timestamp(date), "Product": product, "Category": info["Category"], "Brand": info["Brand"],
        "Warranty": info["Warranty"], "City": city, "SalesPerson": rep, "Designation": designation,
        "Scenario": scenario, "Units": int(units), "Year": pd.Timestamp(date).year,
        "MonthNum": pd.Timestamp(date).month,
    }])


def model_to_bytes(pipe, meta: dict) -> bytes:
    import joblib
    buf = io.BytesIO()
    joblib.dump({"pipeline": pipe, "meta": meta}, buf)
    return buf.getvalue()


# ======================================================================================
# PART 2 – TIME-SERIES FORECASTING
# ======================================================================================
SEASON = 12


def monthly_series(df: pd.DataFrame, metric: str = "Revenue") -> pd.Series:
    """Complete monthly series (gaps filled with 0)."""
    if df.empty:
        return pd.Series(dtype=float)
    s = df.groupby("Month")[metric].sum().sort_index()
    idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    return s.reindex(idx, fill_value=0.0).astype(float)


def _future_index(last: pd.Timestamp, h: int) -> pd.DatetimeIndex:
    return pd.date_range(last + pd.offsets.MonthBegin(1), periods=h, freq="MS")


# ---- individual methods: f(y: np.ndarray, h: int, start_month: int) -> np.ndarray(h)
def fc_seasonal_naive(y, h, start_month=1):
    if len(y) < SEASON:
        return np.repeat(y[-1], h)
    last = y[-SEASON:]
    return np.array([last[i % SEASON] for i in range(h)])


def _month_dummies(months: np.ndarray) -> np.ndarray:
    return np.eye(12)[months - 1]


def fc_trend_seasonal(y, h, start_month=1):
    """OLS on linear trend + month-of-year dummies."""
    n = len(y)
    months = ((np.arange(n) + start_month - 1) % 12) + 1
    X = np.column_stack([np.arange(n), _month_dummies(months)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fm = ((np.arange(n, n + h) + start_month - 1) % 12) + 1
    Xf = np.column_stack([np.arange(n, n + h), _month_dummies(fm)])
    return Xf @ beta


def _hw_run(params, y, m, return_fit=False, h=0):
    a, b, g = params
    n = len(y)
    level = y[:m].mean()
    trend = (y[m:2 * m].mean() - y[:m].mean()) / m if n >= 2 * m else 0.0
    seas = list(y[:m] - level)
    sse = 0.0
    for t in range(n):
        s = seas[t % m] if t < m else seas[-m]
        pred = level + trend + s
        if t >= m:
            sse += (y[t] - pred) ** 2
        new_level = a * (y[t] - s) + (1 - a) * (level + trend)
        trend = b * (new_level - level) + (1 - b) * trend
        seas.append(g * (y[t] - new_level) + (1 - g) * s)
        level = new_level
    if return_fit:
        return np.array([level + (k + 1) * trend + seas[len(seas) - m + (k % m)] for k in range(h)])
    return sse


def fc_holt_winters(y, h, start_month=1):
    """Additive Holt-Winters (level + trend + seasonality), parameters fitted by SSE minimisation."""
    if len(y) < 2 * SEASON:
        return fc_trend_seasonal(y, h, start_month)
    best = minimize(_hw_run, x0=[0.3, 0.05, 0.3], args=(y, SEASON), method="L-BFGS-B",
                    bounds=[(0.01, 0.99), (0.0, 0.5), (0.0, 0.99)])
    return _hw_run(best.x, y, SEASON, return_fit=True, h=h)


def _lag_features(hist: list, t_idx: int, month: int) -> list:
    arr = np.asarray(hist)
    feats = [arr[-1], arr[-2], arr[-3], arr[-6], arr[-12], arr[-3:].mean(), arr[-12:].mean(),
             np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12), t_idx]
    return feats


def fc_ml_boosting(y, h, start_month=1):
    """Gradient boosting on lag / rolling / calendar features, forecasting recursively."""
    if len(y) < 30:
        return fc_trend_seasonal(y, h, start_month)
    X, t = [], []
    for i in range(12, len(y)):
        month = (i + start_month - 1) % 12 + 1
        X.append(_lag_features(list(y[:i]), i, month))
        t.append(y[i])
    X, t = np.array(X), np.array(t)
    # detrend: model the ratio to the 12-month mean so that growth does not fall outside the training range
    base = X[:, 6]
    model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_depth=3, min_samples_leaf=5,
                                          random_state=SEED)
    model.fit(X, t / np.where(base == 0, 1, base))
    hist = list(y)
    out = []
    for k in range(h):
        i = len(hist)
        month = (i + start_month - 1) % 12 + 1
        f = np.array(_lag_features(hist, i, month)).reshape(1, -1)
        pred = float(model.predict(f)[0] * (f[0, 6] if f[0, 6] else 1))
        pred = max(pred, 0.0)
        out.append(pred)
        hist.append(pred)
    return np.array(out)


METHODS = {
    "Seasonal Naive": fc_seasonal_naive,
    "Trend + Seasonality (regression)": fc_trend_seasonal,
    "Holt-Winters (additive)": fc_holt_winters,
    "ML: Gradient Boosting (lags)": fc_ml_boosting,
}
ENSEMBLE = "Ensemble (average of methods)"


def _run(method: str, y: np.ndarray, h: int, start_month: int) -> np.ndarray:
    if method == ENSEMBLE:
        return np.mean([f(y, h, start_month) for f in METHODS.values()], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.clip(METHODS[method](y, h, start_month), 0, None)


def backtest(series: pd.Series, holdout: int = 12, methods: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last ``holdout`` months, forecast them from the earlier history and score each method."""
    methods = methods or list(METHODS) + [ENSEMBLE]
    y = series.to_numpy(float)
    if len(y) < holdout + 18:
        raise ValueError(f"Need at least {holdout + 18} months of history (have {len(y)}).")
    train, test = y[:-holdout], y[-holdout:]
    sm = series.index[0].month
    rows, fcs = [], pd.DataFrame(index=series.index[-holdout:])
    fcs["Actual"] = test
    for m in methods:
        p = _run(m, train, holdout, sm)
        fcs[m] = p
        nz = test != 0
        rows.append({"Method": m,
                     "MAE": float(np.mean(np.abs(test - p))),
                     "RMSE": float(np.sqrt(np.mean((test - p) ** 2))),
                     "MAPE %": float(np.mean(np.abs((test[nz] - p[nz]) / test[nz])) * 100),
                     "Bias %": float((p.sum() - test.sum()) / test.sum() * 100)})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True), fcs


@dataclass
class ForecastResult:
    method: str
    history: pd.Series
    forecast: pd.DataFrame          # index = future months; columns: Forecast, Lower80, Upper80, Lower95, Upper95
    scores: pd.DataFrame
    backtest_fc: pd.DataFrame


def forecast(series: pd.Series, horizon: int = 12, method: str = "auto", holdout: int = 12) -> ForecastResult:
    """Backtest all methods, (optionally) pick the best, refit on the full history and forecast ``horizon`` months."""
    scores, bt = backtest(series, holdout)
    chosen = scores.iloc[0]["Method"] if method == "auto" else method
    y = series.to_numpy(float)
    sm = series.index[0].month
    f = _run(chosen, y, horizon, sm)
    resid = bt["Actual"].to_numpy() - bt[chosen].to_numpy()
    sigma = float(np.sqrt(np.mean(resid ** 2)))
    grow = np.sqrt(1 + 0.04 * np.arange(horizon))   # uncertainty widens slowly with horizon
    out = pd.DataFrame({"Forecast": f}, index=_future_index(series.index[-1], horizon))
    for lvl, z in ((80, 1.2816), (95, 1.96)):
        out[f"Lower{lvl}"] = np.clip(f - z * sigma * grow, 0, None)
        out[f"Upper{lvl}"] = f + z * sigma * grow
    return ForecastResult(chosen, series, out, scores, bt)


def segment_forecast_table(df: pd.DataFrame, dim: str, metric: str, horizon: int = 12) -> pd.DataFrame:
    """Fast forecast per segment (regression with seasonality) vs the same number of recent months."""
    rows = []
    for val, g in df.groupby(dim):
        s = monthly_series(g, metric)
        if len(s) < 24:
            continue
        f = fc_trend_seasonal(s.to_numpy(float), horizon, s.index[0].month).clip(0)
        recent = s.iloc[-horizon:].sum()
        rows.append({dim: val, f"Last {horizon}m": recent, f"Next {horizon}m (forecast)": f.sum(),
                     "Growth %": (f.sum() - recent) / recent * 100 if recent else np.nan})
    return pd.DataFrame(rows).sort_values(f"Next {horizon}m (forecast)", ascending=False).reset_index(drop=True)


def detect_anomalies(series: pd.Series, window: int = 12, z: float = 2.5) -> pd.DataFrame:
    """Flag months deviating from a seasonal-adjusted rolling baseline (robust z-score)."""
    s = series.copy()
    deseason = s - s.groupby(s.index.month).transform("mean") + s.mean()
    base = deseason.rolling(window, center=True, min_periods=3).median()
    resid = deseason - base
    mad = np.median(np.abs(resid - np.median(resid))) or 1.0
    score = 0.6745 * (resid - np.median(resid)) / mad
    return pd.DataFrame({"Value": s, "Score": score, "Anomaly": score.abs() > z})
