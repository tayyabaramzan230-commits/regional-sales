# Sales ML Studio – analysis, visualisation, prediction & forecasting (Streamlit)

An end-to-end machine-learning app for the **Regional Sales** dataset (100,000 order lines · 2015–2020 · 5 cities ·
31 products · 11 sales reps · Actual and Budget scenarios).

| Tab | What it does |
|---|---|
| **Data analysis** | Dataset health (rows, missing, duplicates), column profile, descriptive stats, correlation matrix, distributions + outlier counts, group-by analysis, anomalous-month detection, auto-written insights |
| **Visualization** | Trends (month/quarter/year, split by any dimension), category/city bars, treemap, heat-maps (city×category, year×month), box plots, product bubble chart, price history, rep leaderboard |
| **Predictive modelling** | Train and compare models that predict an order's **Revenue / Profit / Cost**: Unit-Price Trend (hybrid), Gradient Boosting, Random Forest, Extra Trees, Ridge. Time-based or random split, metrics (R², MAE, RMSE, MAPE), actual-vs-predicted, residuals, error by category, permutation feature importance, download predictions and the trained model |
| **Sales predictor** | Enter product, city, sales person, units and date → predicted **revenue, cost, profit, margin**; price-over-time curve (incl. future), what-if order sizes, city comparison, **batch CSV predictions** |
| **Forecasting** | Forecast Revenue / Profit / Units / Orders for the whole company or any segment (city, category, brand, rep, product…) up to 24 months ahead. Four methods + ensemble are back-tested; *Auto* picks the best. 80 % / 95 % intervals, backtest chart, forecast table + CSV, and a forecast for **every segment at once** |

## Quick start (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Open <http://localhost:8501>. `./run.sh` (macOS/Linux) or `run.bat` (Windows) does all of the above.

## Project layout

```
app.py              Streamlit UI (6 tabs)
ml_utils.py         Models, evaluation, feature importance, forecasting methods, backtesting, anomaly detection
sales_utils.py      Data loading/cleaning, filters, KPIs, aggregations, insights
data/               Original Excel file + fast-loading cleaned CSV cache (auto-rebuilt if deleted)
tests/test_ml.py    pytest tests
.streamlit/config.toml, requirements.txt, Dockerfile, docker-compose.yml, render.yaml, Procfile, run.sh, run.bat
```

## How the predictions work

**Order-level model.** Features: product, category, brand, warranty, city, sales person, designation, scenario, units, year, month,
day-of-week, day-of-year. Targets: Revenue, Profit or Cost.

**What we found in this dataset (time-ordered test on the latest 20 % of orders, target = Revenue):**

| Model | R² | MAPE |
|---|---|---|
| Unit-Price Trend (hybrid) | 0.9995 | ≈ 1.3 % |
| Gradient Boosting / Random Forest / Extra Trees | ≈ 0.9967 | ≈ 3.9 % |
| Ridge (linear) | 0.815 | very high |

* Revenue = Units × unit price, and each product's list price depends only on the **year** and rises every year.
  Tree models can only predict price levels they have seen, so they lag on future dates; the hybrid model fits a
  per-product price trend and **extrapolates**, which is why it wins and is the default for the Sales predictor.
* A *random* split lets trees memorise the yearly price list (R² = 1.00) – the app warns about this. Use the time split.
* The ridge regression cannot represent the Units × price interaction – kept as a baseline.

**Forecasting.** The monthly series (72 points) is forecast with seasonal-naive, trend + month-of-year regression,
additive Holt-Winters (parameters fitted with SciPy), and gradient boosting on lag features, plus their average. Each method
is scored on the last *N* months held out (default 12). On total Actual revenue Holt-Winters wins with ≈ 4.8 % MAPE.
Intervals are derived from backtest errors (approximate). Forecasts extrapolate past trend/seasonality only.

## Data notes
* The Excel sheet has blank rows above the header; the loader finds it automatically and renames columns
  (`Sales → Revenue`, `Cost of sales → Cost`, `Units Sold → Units`, `Cities → City`, …). **Profit = Revenue − Cost.**
* No missing values or duplicates. No currency field – the sidebar *Currency label* (default PKR) is display only.
* Parsing the 12 MB workbook takes ~15 s and plenty of RAM, so `data/sales_clean.csv.gz` is cached on first run
  (shipped pre-built). Delete it to force a re-read of the Excel file.
* You can upload another workbook/CSV with the same columns from the sidebar (*Data source*).

## Deployment
* **Streamlit Community Cloud:** push to GitHub → <https://share.streamlit.io> → pick `app.py`. (~1 GB RAM; tree-model training uses a 40k-row sample to stay within it.)
* **Docker:** `docker build -t sales-ml . && docker run -p 8501:8501 sales-ml` (or `docker compose up --build`).
* **Render / Heroku / Railway:** `render.yaml` and `Procfile` are included (they bind to `$PORT`). Give the container ≥ 1 GB RAM.

## Tests
```bash
pip install -r requirements-dev.txt && pytest -q
```

## Troubleshooting
| Symptom | Fix |
|---|---|
| Training Random Forest / Extra Trees is slow | Untick them – the hybrid and Gradient Boosting are fastest and best here |
| "Need at least N months of history" | Choose a broader segment or reduce *Backtest months* |
| Downloaded `.joblib` fails to load | Load it from a folder where `ml_utils.py` is importable (custom model class) |
| Out-of-memory on a small host | Use a ≥ 1 GB instance, or train only the hybrid / Gradient Boosting models |
"# regional-sales" 
"# sales-regional" 
"# sales-ml-studio" 
"# sales-ml-studio" 
"# regional-sales" 
"# regional-sales" 
