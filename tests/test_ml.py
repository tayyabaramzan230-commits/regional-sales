"""Tests for the ML + forecasting layer.  Run:  pytest -q"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

import ml_utils as ml
import sales_utils as su


@pytest.fixture(scope="module")
def actual():
    d = su.load_data()
    return d[d["Scenario"] == "Actual"]


def test_hybrid_model_time_split(actual):
    r = ml.train_models(actual, "Revenue", ["Unit-Price Trend (hybrid)", "Ridge (linear)"], "time")
    assert r.best_model == "Unit-Price Trend (hybrid)"
    m = r.metrics.set_index("Model")
    assert m.loc["Unit-Price Trend (hybrid)", "R²"] > 0.99
    assert m.loc["Ridge (linear)", "R²"] < m.loc["Unit-Price Trend (hybrid)", "R²"]


def test_gradient_boosting_trains(actual):
    r = ml.train_models(actual.sample(8000, random_state=1), "Profit", ["Gradient Boosting"], "random")
    assert r.metrics.iloc[0]["R²"] > 0.9


def test_single_prediction_scales_with_units(actual):
    lk = ml.product_lookup(actual)
    pipe = ml.make_model("Unit-Price Trend (hybrid)").fit(ml.add_features(actual)[ml.ALL_FEATURES], actual["Revenue"])
    one = ml.build_order_row(lk, lk.index[0], "Karachi", "Faisal Ameen", "Area manager", 1, "2021-03-01")
    three = ml.build_order_row(lk, lk.index[0], "Karachi", "Faisal Ameen", "Area manager", 3, "2021-03-01")
    p1, p3 = ml.predict_orders(pipe, one)[0], ml.predict_orders(pipe, three)[0]
    assert p3 == pytest.approx(3 * p1, rel=1e-6) and p1 > 0


def test_forecast_shapes(actual):
    s = ml.monthly_series(actual, "Revenue")
    assert len(s) == 72
    fr = ml.forecast(s, 12)
    assert len(fr.forecast) == 12 and fr.forecast.index[0] == pd.Timestamp("2021-01-01")
    assert (fr.forecast["Upper95"] >= fr.forecast["Forecast"]).all()
    assert (fr.forecast["Lower95"] <= fr.forecast["Forecast"]).all()
    assert fr.scores["RMSE"].is_monotonic_increasing


def test_each_method_runs(actual):
    y = ml.monthly_series(actual, "Revenue").to_numpy()
    for name, f in ml.METHODS.items():
        out = f(y, 6, 1)
        assert out.shape == (6,) and np.isfinite(out).all(), name


def test_short_history_raises(actual):
    with pytest.raises(ValueError):
        ml.forecast(ml.monthly_series(actual, "Revenue").iloc[:20], 6)


def test_segment_table(actual):
    t = ml.segment_forecast_table(actual, "Category", "Revenue", 12)
    assert len(t) == 6
