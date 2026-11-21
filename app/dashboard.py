"""
Sri Lanka Dengue Surveillance Dashboard

Run from the project folder:
    streamlit run app/dashboard.py

Uses the files made by the notebooks:
    dataset/processed/01_validated_weekly_data.csv  original report dates (notebook 01)
    dataset/processed/02_cleaned_weekly_data.csv   weekly cases per reporting area (notebook 02)
    dataset/processed/03_risk_thresholds.csv       district risk thresholds (notebook 03)
    dataset/processed/test.csv                     test weeks 2024-2026 (notebook 03)
    models/05_random_forest_results.json           Random Forest settings and validation scores (notebook 05, bagging)
    models/05_final_random_forest_model.joblib     final trained Random Forest model (notebook 05)
    models/06_xgboost_results.json                 XGBoost settings and validation scores (notebook 06, boosting)
    models/06_final_xgboost_model.json             final trained XGBoost model (notebook 06)
    models/07_final_test_results.json              permutation importance, sensitivity check (notebook 07)
    models/08_hybrid_rf_arima_results.json         optional: RF + ARIMA experiment (notebook 08)
"""
import datetime
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
VALIDATED_PATH = ROOT / "dataset" / "processed" / "01_validated_weekly_data.csv"
DATA_PATH = ROOT / "dataset" / "processed" / "02_cleaned_weekly_data.csv"
THRESH_PATH = ROOT / "dataset" / "processed" / "03_risk_thresholds.csv"
TEST_PATH = ROOT / "dataset" / "processed" / "test.csv"
RF_RESULTS_PATH = ROOT / "models" / "05_random_forest_results.json"
RF_MODEL_PATH = ROOT / "models" / "05_final_random_forest_model.joblib"
XGB_RESULTS_PATH = ROOT / "models" / "06_xgboost_results.json"
XGB_MODEL_PATH = ROOT / "models" / "06_final_xgboost_model.json"
RESULTS_07_PATH = ROOT / "models" / "07_final_test_results.json"
RESULTS_08_PATH = ROOT / "models" / "08_hybrid_rf_arima_results.json"

ENSEMBLES = ["Random Forest", "XGBoost"]

st.set_page_config(page_title="Sri Lanka Dengue Dashboard", layout="wide")

# chart colours (models use the same colours as notebooks 04-06)
MODEL_COLORS = {"Persistence": "#898781", "Decision Tree": "#eb6834", "Random Forest": "#2a78d6", "XGBoost": "#1baf7a"}
BAR_COLOR = "#2a78d6"
RISK_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}

dark_mode = st.context.theme.type == "dark"
INK = "#f0efec" if dark_mode else "#0b0b0b"         # actual case counts and text
MUTED_INK = "#c3c2b7" if dark_mode else "#52514e"
GRID = "#2c2c2a" if dark_mode else "#e1e0d9"

DISCLAIMER = ("Risk levels are **relative statistical levels** for each reporting area, based on its own past weekly "
              "case counts. They are **not official outbreak alerts** from the Ministry of Health.")


# ---------------------------------------------------------------- loading data and models

@st.cache_data
def load_weekly_data():
    return pd.read_csv(DATA_PATH, parse_dates=["start_date", "end_date"])


@st.cache_data
def load_report_dates():
    """Original report dates of every (year, week) before notebook 02 re-spaced the weeks."""
    validated = pd.read_csv(VALIDATED_PATH, parse_dates=["start_date", "end_date"])
    return validated.drop_duplicates(["year", "week"]).set_index(["year", "week"])[["start_date", "end_date"]]


@st.cache_data
def load_thresholds():
    return pd.read_csv(THRESH_PATH, index_col="district")


@st.cache_resource
def load_models():
    """The two final ensemble models, both trained on target weeks 2007-2023."""
    xgb = XGBRegressor()
    xgb.load_model(XGB_MODEL_PATH)       # notebook 06 (boosting)
    rf = joblib.load(RF_MODEL_PATH)      # notebook 05 (bagging; saved with n_jobs=1 for repeatable predictions)
    return {"Random Forest": rf, "XGBoost": xgb}


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


rf_results = json.loads(RF_RESULTS_PATH.read_text())     # notebook 05 (bagging)
xgb_results = json.loads(XGB_RESULTS_PATH.read_text())   # notebook 06 (boosting)
for key in ["features", "target", "validation_year"]:
    assert rf_results[key] == xgb_results[key], key
config = {
    "features": rf_results["features"],
    "target": rf_results["target"],
    "validation_year": rf_results["validation_year"],
    # persistence and Decision Tree are in both files (same values); Random Forest from 05, XGBoost from 06
    "validation_scores": {**rf_results["validation_scores"], **xgb_results["validation_scores"]},
}
results_07 = load_json(RESULTS_07_PATH)
results_08 = load_json(RESULTS_08_PATH)
FEATURES = config["features"]

weekly = load_weekly_data()
report_dates = load_report_dates()
thresholds = load_thresholds()
models = load_models()


def predict_with(model_name, X):
    """Predictions of one ensemble model; negative values are clipped to 0 (as in notebooks 05-07)."""
    return np.clip(models[model_name].predict(X[FEATURES]), 0, None)


# ---------------------------------------------------------------- features, prediction and risk

@st.cache_data
def add_features(weekly_data):
    """Build the same 11 input features as notebook 03 for every district-week."""
    df = weekly_data.sort_values(["district", "start_date"]).reset_index(drop=True)
    district_cases = df.groupby("district")["cases"]

    df["current_week_cases"] = df["cases"]
    df["previous_week_cases"] = district_cases.shift(1)
    df["previous_2_weeks_cases"] = district_cases.shift(2)
    df["previous_3_weeks_cases"] = district_cases.shift(3)
    df["seasonal_cases_previous_52_weeks"] = district_cases.shift(52)
    df["moving_average_4_weeks"] = district_cases.transform(lambda x: x.rolling(window=4).mean())
    df["moving_average_8_weeks"] = district_cases.transform(lambda x: x.rolling(window=8).mean())
    df["weekly_case_change"] = df["current_week_cases"] - df["previous_week_cases"]

    day_of_year = df["start_date"].dt.dayofyear
    days_in_year = np.where(df["start_date"].dt.is_leap_year, 366, 365)
    year_position = (day_of_year - 1) / days_in_year
    df["seasonal_week_sin"] = np.sin(2 * np.pi * year_position)
    df["seasonal_week_cos"] = np.cos(2 * np.pi * year_position)
    return df


def risk_level(cases, district):
    """Low / Medium / High using the district thresholds from notebook 03
    (75th and 90th percentile of the district's weekly cases, 2007-2023)."""
    if cases > thresholds.loc[district, "high_above"]:
        return "High"
    if cases > thresholds.loc[district, "medium_above"]:
        return "Medium"
    return "Low"


def predict_next_week(model_name):
    """Predict next week's cases for every reporting area from its latest reported week."""
    latest = add_features(weekly).groupby("district").tail(1).copy()
    for name in ENSEMBLES:
        latest[name] = np.round(predict_with(name, latest)).astype(int)
    latest["predicted_cases"] = latest[model_name]
    latest["target_date"] = latest["start_date"] + pd.Timedelta(weeks=1)
    latest["risk"] = [risk_level(c, d) for c, d in zip(latest["predicted_cases"], latest["district"])]
    return latest.set_index("district")


@st.cache_data
def test_period_predictions():
    """Final test (target weeks 2024-2026), the same rows as notebooks 05-07:
    actual cases and the predictions of persistence, Random Forest and XGBoost."""
    test = pd.read_csv(TEST_PATH, parse_dates=["start_date", "target_date"])
    test = test[~(test["is_imputed"] | test["target_imputed"])].reset_index(drop=True)  # leave out interpolated weeks
    test["actual"] = test[config["target"]]
    test["Persistence"] = test["current_week_cases"].astype(float)
    for name in ENSEMBLES:
        test[name] = predict_with(name, test)
    return test


def score_table(test, rows=None):
    """MAE and RMSE of every model on the test rows, plus the change compared with persistence."""
    part = test if rows is None else test[rows]
    table = pd.DataFrame({name: {"MAE": mean_absolute_error(part["actual"], part[name]),
                                 "RMSE": root_mean_squared_error(part["actual"], part[name])}
                          for name in TEST_MODELS}).T
    table["MAE vs persistence (%)"] = 100 * (table["MAE"] / table.loc["Persistence", "MAE"] - 1)
    table["RMSE vs persistence (%)"] = 100 * (table["RMSE"] / table.loc["Persistence", "RMSE"] - 1)
    return table


TEST_MODELS = ["Persistence", "Random Forest", "XGBoost"]


# ---------------------------------------------------------------- chart helpers

def style_chart(fig, y_title, height=380):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=80, b=10),  # room for the title above the legend
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        barcornerradius=4,
        yaxis_title=y_title,
    )
    fig.update_yaxes(rangemode="tozero", gridcolor=GRID, zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def add_threshold_lines(fig, district):
    for level, column in [("Medium", "medium_above"), ("High", "high_above")]:
        value = thresholds.loc[district, column]
        fig.add_hline(y=value, line_dash="dash", line_width=1.5, line_color=RISK_COLORS[level],
                      annotation_text=f"{level} risk above {value:.0f}", annotation_position="top left",
                      annotation_font_color=MUTED_INK)


def show_risk(risk):
    message = f"**{risk} risk** (relative statistical risk level, not an official outbreak alert)"
    {"Low": st.success, "Medium": st.warning, "High": st.error}[risk](message)


# ---------------------------------------------------------------- sidebar

districts = sorted(weekly["district"].unique())
latest_date = weekly["start_date"].max()
latest_rows = weekly[weekly["start_date"] == latest_date]
latest_year, latest_week_no = int(latest_rows["year"].iloc[0]), int(latest_rows["week"].iloc[0])
forecast_date = latest_date + pd.Timedelta(weeks=1)
source_start, source_end = report_dates.loc[(latest_year, latest_week_no)]

st.sidebar.title("Dengue Dashboard")
district = st.sidebar.selectbox("Reporting area", districts, index=districts.index("Colombo"))
st.sidebar.caption("26 reporting areas: the 25 districts plus Kalmunai (spelt *Kalmune* in the data), which is "
                   "reported separately. The selection is used in the *Historical trends*, *Prediction & risk* "
                   "and *Model comparison* tabs.")
st.sidebar.divider()
st.sidebar.markdown(f"**Latest data:** {latest_year} week {latest_week_no}, week starting {latest_date:%d %b %Y}")
st.sidebar.caption(f"In the source report this week is dated {source_start:%d %b} - {source_end:%d %b %Y}. "
                   "Notebook 02 re-spaced all weeks to exactly 7 days, which moved some 2026 dates by 1-2 days.")
st.sidebar.markdown("**Models:** Random Forest and XGBoost, both trained on target weeks 2007-2023")
st.sidebar.caption("Data: Weekly Epidemiological Reports, Epidemiology Unit, Ministry of Health, Sri Lanka, "
                   "compiled in the denguedatahub repository (Dr. Thiyanga S. Talagala).")

st.title("Sri Lanka Dengue Surveillance Dashboard")

today = datetime.date.today()
if today > (forecast_date + pd.Timedelta(days=6)).date():
    st.warning(f"**The dataset ends with the week starting {latest_date:%d %b %Y}.** All figures and the "
               f"next-week prediction refer to that time (the prediction is for the week starting "
               f"{forecast_date:%d %b %Y}), not to the current week (today is {today:%d %b %Y}).")

tab_overview, tab_trends, tab_prediction, tab_models = st.tabs(
    ["Overview", "Historical trends", "Prediction & risk", "Model comparison"])


# ---------------------------------------------------------------- tab 1: national overview

with tab_overview:
    st.subheader("National dengue statistics")

    national = weekly.groupby("start_date", as_index=False)["cases"].sum()
    this_week, last_week = national["cases"].iloc[-1], national["cases"].iloc[-2]

    so_far = weekly[weekly["year"] == latest_year]["cases"].sum()
    same_weeks_last_year = weekly[(weekly["year"] == latest_year - 1) & (weekly["week"] <= latest_week_no)]["cases"].sum()
    top_district = latest_rows.loc[latest_rows["cases"].idxmax()]

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Cases, week starting {latest_date:%d %b %Y}", f"{this_week:,}",
                delta=f"{this_week - last_week:+,} vs previous week", delta_color="inverse")
    col2.metric(f"Cases in {latest_year} so far (weeks 1-{latest_week_no})", f"{so_far:,}",
                delta=f"{so_far - same_weeks_last_year:+,} vs same weeks of {latest_year - 1}", delta_color="inverse")
    col3.metric("Most cases in the latest week (reporting area)", top_district["district"],
                delta=f"{top_district['cases']:,} cases",
                delta_color="off")

    fig = go.Figure(go.Scatter(x=national["start_date"], y=national["cases"], mode="lines", name="National cases",
                               line=dict(color=INK, width=2),
                               hovertemplate="%{y:,} cases<extra></extra>"))
    fig.update_layout(title="Weekly dengue cases, all 26 reporting areas")
    st.plotly_chart(style_chart(fig, "cases per week"), width="stretch")

    left, right = st.columns(2)

    with left:
        yearly = weekly[weekly["year"] >= 2007].groupby("year", as_index=False)["cases"].sum()
        fig = go.Figure(go.Bar(x=yearly["year"], y=yearly["cases"], marker_color=BAR_COLOR, name="Yearly cases",
                               hovertemplate="%{x}: %{y:,} cases<extra></extra>"))
        fig.update_layout(title="Total cases per year", hovermode="closest")
        st.plotly_chart(style_chart(fig, "cases per year"), width="stretch")
        st.caption(f"{latest_year} is a partial year (weeks 1-{latest_week_no}).")

    with right:
        by_district = latest_rows.sort_values("cases")
        fig = go.Figure(go.Bar(x=by_district["cases"], y=by_district["district"], orientation="h",
                               marker_color=BAR_COLOR, name="Cases",
                               hovertemplate="%{y}: %{x:,} cases<extra></extra>"))
        fig.update_layout(title=f"Cases by reporting area, week starting {latest_date:%d %b %Y}", hovermode="closest")
        style_chart(fig, None, height=560)
        fig.update_xaxes(showgrid=True, gridcolor=GRID)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch")

    with st.expander("Show data: national weekly cases"):
        st.dataframe(national.sort_values("start_date", ascending=False)
                     .rename(columns={"start_date": "week starting", "cases": "national cases"}),
                     hide_index=True, width="stretch")


# ---------------------------------------------------------------- tab 2: district historical trends

with tab_trends:
    st.subheader(f"Weekly dengue cases in {district}")

    first_year, last_year = int(weekly["year"].min()), int(weekly["year"].max())
    start_year, end_year = st.slider("Years", first_year, last_year, (last_year - 4, last_year))

    history = weekly[(weekly["district"] == district) & weekly["year"].between(start_year, end_year)]
    estimated = history[history["is_imputed"]]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total cases in these years", f"{history['cases'].sum():,}")
    col2.metric("Average cases per week", f"{history['cases'].mean():.1f}")
    peak = history.loc[history["cases"].idxmax()]
    col3.metric("Highest week", f"{peak['cases']:,} cases", delta=f"week starting {peak['start_date']:%d %b %Y}",
                delta_color="off")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["start_date"], y=history["cases"], mode="lines", name="Weekly cases",
                             line=dict(color=INK, width=2), hovertemplate="%{y:,} cases<extra></extra>"))
    if len(estimated) > 0:
        fig.add_trace(go.Scatter(x=estimated["start_date"], y=estimated["cases"], mode="markers",
                                 name="Estimated week (missing in source, interpolated)",
                                 marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(color=INK, width=2)),
                                 hovertemplate="estimated<extra></extra>"))
    add_threshold_lines(fig, district)
    fig.update_layout(title=f"{district}, {start_year}-{end_year}")
    st.plotly_chart(style_chart(fig, "cases per week", height=420), width="stretch")
    st.caption("Dashed lines are the risk thresholds of this reporting area. " + DISCLAIMER)

    with st.expander(f"Show data: {district} weekly cases"):
        st.dataframe(history[["year", "week", "start_date", "cases", "is_imputed"]]
                     .sort_values("start_date", ascending=False)
                     .rename(columns={"start_date": "week starting", "is_imputed": "estimated"}),
                     hide_index=True, width="stretch")


# ---------------------------------------------------------------- tab 3: next-week prediction and risk

with tab_prediction:
    model_name = st.radio("Prediction model", ENSEMBLES, horizontal=True,
                          help="Both models are trained on target weeks 2007-2023. "
                               "See the Model comparison tab for how each performed on 2024-2026.")
    forecast = predict_next_week(model_name)
    row = forecast.loc[district]
    other_model = [name for name in ENSEMBLES if name != model_name][0]

    st.subheader(f"Next-week prediction for {district}")
    st.caption(f"Predicted with the final {model_name} model from the latest week in the dataset "
               f"(starting {row['start_date']:%d %b %Y}). For comparison, {other_model} predicts "
               f"{row[other_model]:,} cases. This is a forecast for the week starting {row['target_date']:%d %b %Y}, "
               "not for the current week.")

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Reported cases, week starting {row['start_date']:%d %b}", f"{row['cases']:,}")
    change = row["predicted_cases"] - row["cases"]
    col2.metric(f"Predicted cases, week starting {row['target_date']:%d %b}", f"{row['predicted_cases']:,}",
                delta=f"{change:+,} vs latest week" if change != 0 else "same as latest week",
                delta_color="inverse" if change != 0 else "off")
    with col3:
        st.markdown("**Risk level for next week**")
        show_risk(row["risk"])

    recent = weekly[weekly["district"] == district].tail(12)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent["start_date"], y=recent["cases"], mode="lines+markers", name="Reported cases",
                             line=dict(color=INK, width=2), marker=dict(size=8),
                             hovertemplate="reported: %{y:,}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[row["start_date"], row["target_date"]], y=[row["cases"], row["predicted_cases"]],
                             mode="lines+markers", name=f"{model_name} prediction",
                             line=dict(color=MODEL_COLORS[model_name], width=2, dash="dot"),
                             marker=dict(size=[0, 11], color=MODEL_COLORS[model_name]),
                             hovertemplate="predicted: %{y:,}<extra></extra>"))
    add_threshold_lines(fig, district)
    fig.update_layout(title=f"{district}: last 12 reported weeks and next-week prediction")
    st.plotly_chart(style_chart(fig, "cases per week"), width="stretch")

    medium_above = thresholds.loc[district, "medium_above"]
    high_above = thresholds.loc[district, "high_above"]
    st.markdown(f"""
**How the risk level is set for {district}** (thresholds from 2007-2023 weekly cases):

| Risk level | Predicted weekly cases | Meaning |
|---|---|---|
| Low | {medium_above:.0f} or fewer | a normal week for this district (75% of past weeks were at or below this) |
| Medium | more than {medium_above:.0f}, up to {high_above:.0f} | busier than 75% of past weeks |
| High | more than {high_above:.0f} | busier than 90% of past weeks |
""")
    st.info(DISCLAIMER)
    st.caption("The most recent weeks of surveillance data can be incomplete because some reports arrive late, "
               "so the prediction may be too low when recent counts are under-reported.")

    st.divider()
    st.subheader("Risk classification for all 26 reporting areas")
    st.caption(f"Predicted with {model_name} for the week starting {forecast_date:%d %b %Y}.")

    counts = forecast["risk"].value_counts()
    col1, col2, col3 = st.columns(3)
    for col, level in zip([col1, col2, col3], ["High", "Medium", "Low"]):
        n = counts.get(level, 0)
        col.metric(f"{level} risk", f"{n} reporting area" if n == 1 else f"{n} reporting areas")

    table = forecast.reset_index().join(thresholds, on="district")
    table = table.sort_values(["predicted_cases"], ascending=False)
    st.dataframe(
        table[["district", "cases", "predicted_cases", "risk", "medium_above", "high_above"]].rename(columns={
            "district": "reporting area",
            "cases": f"reported (week of {latest_date:%d %b})",
            "predicted_cases": f"predicted next week ({model_name})",
            "medium_above": "Medium above",
            "high_above": "High above",
        }),
        hide_index=True, width="stretch", height=950,
        column_config={"Medium above": st.column_config.NumberColumn(format="%.0f"),
                       "High above": st.column_config.NumberColumn(format="%.0f")},
    )


# ---------------------------------------------------------------- tab 4: model comparison

with tab_models:
    st.subheader("Random Forest (bagging) vs XGBoost (boosting)")
    st.markdown("""
Both models are **ensembles of decision trees** (a single tree asks a series of yes/no questions about the
features). They combine their trees in opposite ways:

| | Random Forest: bagging | XGBoost: boosting |
|---|---|---|
| How the trees are built | independently, each on a bootstrap sample (rows drawn at random with replacement), with a random subset of features at each split | one after another; each new tree is fitted to the errors the earlier trees still make |
| How predictions are combined | **average** of all trees | **sum** of all trees, each scaled down by the learning rate |
| Main effect | reduces **variance**: averaging cancels the overfitting of single deep trees | reduces **bias**: keeps correcting what is still wrong, but can over-fit rare patterns |
| Prediction range | always within the range of past training values | can go beyond it (and below 0, so predictions are clipped) |
| Settings used here | 200 trees, at least 10 rows per leaf | 400 rounds, tree depth 2, learning rate 0.05 |

**Persistence** (next week's cases = this week's cases) is the baseline that every model must beat. A **Decision
Tree** (one tree, depth 5) is shown on the validation year as the single-tree reference. All models use the same 11
features, the same leakage check and the same chronological split.
""")

    scores = pd.DataFrame(config["validation_scores"]).T
    scores = scores.loc[["Persistence", "Decision Tree", "Random Forest", "XGBoost"]]
    scores["MAE vs persistence (%)"] = 100 * (scores["MAE"] / scores.loc["Persistence", "MAE"] - 1)
    scores["RMSE vs persistence (%)"] = 100 * (scores["RMSE"] / scores.loc["Persistence", "RMSE"] - 1)

    st.markdown(f"#### 1. Validation year {config['validation_year']} (used to choose the settings)")
    st.markdown(f"Trained on target weeks 2007-{config['validation_year'] - 1}, scored on every district-week of "
                f"{config['validation_year']}. Lower is better.")

    col1, col2 = st.columns(2)
    for col, metric in zip([col1, col2], ["MAE", "RMSE"]):
        fig = go.Figure(go.Bar(x=scores.index, y=scores[metric],
                               marker_color=[MODEL_COLORS[name] for name in scores.index],
                               text=[f"{v:.2f}" for v in scores[metric]], textposition="outside",
                               textfont_color=INK, hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
        fig.add_hline(y=scores.loc["Persistence", metric], line_dash="dash", line_width=1, line_color=MUTED_INK)
        fig.update_layout(title=f"{metric} (dashed line = persistence)", hovermode="closest", showlegend=False)
        style_chart(fig, "cases per district-week")
        fig.update_yaxes(range=[0, scores[metric].max() * 1.15])
        col.plotly_chart(fig, width="stretch")

    st.dataframe(scores.round(2), width="stretch")
    st.caption("MAE = average size of the error, in cases per district-week. "
               "RMSE = like MAE, but large errors count more. Negative % = smaller error than persistence.")

    valid_rf, valid_xgb = scores.loc["Random Forest", "MAE"], scores.loc["XGBoost", "MAE"]
    valid_best = "Random Forest" if valid_rf < valid_xgb else "XGBoost"
    st.markdown(f"On {config['validation_year']}, **{valid_best}** had the slightly lower MAE "
                f"(XGBoost {valid_xgb:.2f}, Random Forest {valid_rf:.2f}: a difference of "
                f"{abs(valid_rf - valid_xgb):.2f} cases per district-week). "
                "A difference this small on a single year is not enough to call one model better, so both were "
                "evaluated on the untouched test set below.")

    # ------------------------------------------------ final test
    st.divider()
    test = test_period_predictions()
    test_scores = score_table(test)
    st.markdown("#### 2. Final test 2024-2026 (unseen data)")
    st.markdown(f"Both ensembles were retrained on target weeks 2007-2023 with the settings fixed on "
                f"{config['validation_year']} and scored once on the **same {len(test):,} district-weeks** "
                f"(target weeks {test['target_date'].min():%d %b %Y} to {test['target_date'].max():%d %b %Y}). "
                "Nothing was tuned on these weeks.")

    col1, col2 = st.columns(2)
    for col, metric in zip([col1, col2], ["MAE", "RMSE"]):
        fig = go.Figure(go.Bar(x=TEST_MODELS, y=test_scores[metric],
                               marker_color=[MODEL_COLORS[name] for name in TEST_MODELS],
                               text=[f"{v:.2f}" for v in test_scores[metric]], textposition="outside",
                               textfont_color=INK, hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
        fig.add_hline(y=test_scores.loc["Persistence", metric], line_dash="dash", line_width=1, line_color=MUTED_INK)
        fig.update_layout(title=f"Test {metric} (dashed line = persistence)", hovermode="closest", showlegend=False)
        style_chart(fig, "cases per district-week")
        fig.update_yaxes(range=[0, test_scores[metric].max() * 1.15])
        col.plotly_chart(fig, width="stretch")

    st.dataframe(test_scores.round(2), width="stretch")

    year_rows = []
    for year, part in test.groupby(test["target_date"].dt.year):
        year_rows.append({"test year": str(year), "district-weeks": len(part), "mean weekly cases": part["actual"].mean(),
                          **{f"{name} MAE": mean_absolute_error(part["actual"], part[name]) for name in TEST_MODELS}})
    by_year = pd.DataFrame(year_rows)
    st.markdown("**Test MAE by year** (2026 has data until August)")
    st.dataframe(by_year.round(2), hide_index=True, width="stretch")

    # ------------------------------------------------ actual vs predicted
    st.divider()
    st.markdown(f"#### 3. Actual vs predicted: {district}, test period")
    part = test[test["district"] == district]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=part["target_date"], y=part["actual"], mode="lines", name="Actual",
                             line=dict(color=INK, width=2.5), hovertemplate="actual: %{y:,}<extra></extra>"))
    for name in TEST_MODELS:
        fig.add_trace(go.Scatter(x=part["target_date"], y=part[name], mode="lines", name=name,
                                 line=dict(color=MODEL_COLORS[name], width=1.8),
                                 hovertemplate=f"{name}: %{{y:,.0f}}<extra></extra>"))
    fig.update_layout(title=f"{district}: weekly cases, actual vs one-week-ahead predictions")
    st.plotly_chart(style_chart(fig, "cases per week", height=420), width="stretch")
    area_scores = score_table(test, test["district"] == district)
    st.caption(f"MAE in {district}: " + ", ".join(f"{name} {area_scores.loc[name, 'MAE']:.2f}" for name in TEST_MODELS)
               + ". Click a legend entry to hide or show a line.")

    # ------------------------------------------------ feature importance
    st.divider()
    st.markdown("#### 4. Feature importance")

    def importance_chart(table, title, x_title):
        order = table.sort_values("Random Forest").index
        fig = go.Figure()
        for name in ENSEMBLES:
            fig.add_trace(go.Bar(y=order, x=table.loc[order, name], orientation="h", name=name,
                                 marker_color=MODEL_COLORS[name], hovertemplate=f"{name}: %{{x:.3f}}<extra></extra>"))
        fig.update_layout(title=title, barmode="group", hovermode="closest")
        style_chart(fig, None, height=480)
        fig.update_xaxes(title=x_title, showgrid=True, gridcolor=GRID)
        fig.update_yaxes(showgrid=False)
        return fig

    xgb_gain = pd.Series(models["XGBoost"].get_booster().get_score(importance_type="gain")).reindex(FEATURES).fillna(0)
    builtin = pd.DataFrame({"Random Forest": models["Random Forest"].feature_importances_,
                            "XGBoost": xgb_gain / xgb_gain.sum()}, index=FEATURES)

    col1, col2 = st.columns(2)
    col1.plotly_chart(importance_chart(builtin, "Built-in importance (training data)", "share of total importance"),
                      width="stretch")
    if results_07:
        permutation = pd.DataFrame(results_07["feature_importance"]).T.rename(columns={
            "Random Forest (test MAE increase)": "Random Forest", "XGBoost (test MAE increase)": "XGBoost"})
        col2.plotly_chart(importance_chart(permutation[ENSEMBLES], "Permutation importance (test set)",
                                           "test MAE increase when shuffled (cases)"),
                          width="stretch")
    st.markdown("""
**How to read these charts (cautions)**
- *Built-in importance* is the Random Forest's **impurity** importance and XGBoost's **gain**. Both are measured on
  the training data, are on different scales, and favour features with many distinct values; compare the **ranking**,
  not the numbers.
- *Permutation importance* (notebook 07) shuffles one feature in the test data and measures how much the error grows.
  It is measured on unseen data and is usually more reliable.
- The lag and moving-average features are strongly **correlated**, so they share importance: a feature can look
  unimportant because another one carries the same information.
- Importance shows what a model **uses**, not what **causes** dengue cases.
""")

    # ------------------------------------------------ interpretation
    st.divider()
    st.markdown("#### 5. Interpretation (based only on the measured results)")

    rf_scores, xgb_scores = test_scores.loc["Random Forest"], test_scores.loc["XGBoost"]
    lines = []
    if rf_scores["MAE"] < xgb_scores["MAE"] and rf_scores["RMSE"] < xgb_scores["RMSE"]:
        winner, loser = "Random Forest", "XGBoost"
    elif xgb_scores["MAE"] < rf_scores["MAE"] and xgb_scores["RMSE"] < rf_scores["RMSE"]:
        winner, loser = "XGBoost", "Random Forest"
    else:
        winner = loser = None

    if winner:
        lines.append(f"- On the unseen 2024-2026 weeks, **{winner} generalised better than {loser}**: lower MAE "
                     f"({test_scores.loc[winner, 'MAE']:.2f} vs {test_scores.loc[loser, 'MAE']:.2f}) and lower RMSE "
                     f"({test_scores.loc[winner, 'RMSE']:.2f} vs {test_scores.loc[loser, 'RMSE']:.2f}).")
        if winner != valid_best:
            lines.append(f"- This reverses the {config['validation_year']} validation ranking, where {valid_best} "
                         "was very slightly ahead: the small validation lead did not carry over to new years.")
    else:
        lines.append("- MAE and RMSE rank the two ensembles differently, so neither generalised clearly better.")

    for name in ENSEMBLES:
        mae_change = test_scores.loc[name, "MAE vs persistence (%)"]
        rmse_change = test_scores.loc[name, "RMSE vs persistence (%)"]
        if mae_change < 0 and rmse_change < 0:
            verdict = "beats persistence on both metrics"
        elif mae_change < 0 or rmse_change < 0:
            verdict = "does not clearly beat persistence (better on one metric, worse on the other)"
        else:
            verdict = "does not beat persistence"
        better_years = [r["test year"] for _, r in by_year.iterrows() if r[f"{name} MAE"] < r["Persistence MAE"]]
        worse_years = [r["test year"] for _, r in by_year.iterrows() if r[f"{name} MAE"] >= r["Persistence MAE"]]
        lines.append(f"- **{name}** vs persistence: MAE {mae_change:+.1f}%, RMSE {rmse_change:+.1f}%, so it {verdict}. "
                     f"Lower MAE than persistence in: {', '.join(better_years) or 'none'}; "
                     f"not in: {', '.join(worse_years) or 'none'}.")

    lines.append("- The largest errors of both ensembles are at the peaks of the 2026 outbreak, where they predicted "
                 "too many cases after the peak. XGBoost overshot most, because it learned from a few extreme 2017 "
                 "weeks that a large weekly jump is followed by an even higher week (notebook 07, section 5).")
    lines.append("- The results come from one test period dominated by one large outbreak, without a statistical "
                 "significance test, so small differences should not be over-interpreted.")
    st.markdown("\n".join(lines))

    if results_07:
        with st.expander("Data-quality check: duplicated weeks (notebook 07, section 7)"):
            sensitivity = results_07["sensitivity_without_duplicated_weeks"]
            st.markdown(f"Notebook 02 found pairs of consecutive weeks with identical counts in all 26 reporting areas "
                        f"(probably the same report copied twice). Two pairs fall in the test period, where persistence "
                        f"gets a perfect score on the copied week. The table re-scores the same predictions without the "
                        f"rows that use a duplicated week ({sensitivity['rows']:,} of {len(test):,} rows remain). "
                        "The official test results above include all rows.")
            st.dataframe(pd.DataFrame(sensitivity["scores"]).T.loc[TEST_MODELS].round(2), width="stretch")

    if results_08:
        with st.expander("Experimental extension: Random Forest + ARIMA hybrid (notebook 08)"):
            st.markdown("An ARIMA(1,0,0) model predicts next week's Random Forest error from its recent errors, and the "
                        "hybrid adds that correction to the Random Forest prediction. ARIMA was fitted only on "
                        "out-of-sample 2023 errors and applied week by week on the test set using only errors already "
                        "known at that time. The hybrid is **not** used for the predictions in this dashboard.")
            st.dataframe(pd.DataFrame(results_08["test_scores"]).T.round(2), width="stretch")
