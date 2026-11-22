"""
Sri Lanka Dengue Dashboard: bagging vs boosting vs stacking

Run from the project folder:
    streamlit run app/dashboard.py

Uses the files made by the notebooks:
    dataset/processed/02_cleaned_weekly_data.csv   weekly cases per reporting area (notebook 02)
    dataset/processed/03_risk_thresholds.csv       district risk thresholds (notebook 03)
    dataset/processed/test.csv                     test weeks 2024-2026 (notebook 03)
    models/05_random_forest_results.json           Random Forest settings (notebook 05, bagging)
    models/05_final_random_forest_model.joblib     final Random Forest model (notebook 05)
    models/06_xgboost_results.json                 XGBoost settings + GBM / CatBoost scores (notebook 06, boosting)
    models/06_final_xgboost_model.json             final XGBoost model (notebook 06)
    models/07_stacking_results.json                stacking meta-model and its weights (notebook 07)
    models/07_final_linear_regression_model.joblib Linear Regression base model of the stack (notebook 07)
    models/07_stacking_meta_model.joblib           stacking meta-model (notebook 07)
    models/08_final_test_results.json              permutation importance (notebook 08)
"""
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
DATA_PATH = ROOT / "dataset" / "processed" / "02_cleaned_weekly_data.csv"
THRESH_PATH = ROOT / "dataset" / "processed" / "03_risk_thresholds.csv"
TEST_PATH = ROOT / "dataset" / "processed" / "test.csv"
RF_RESULTS_PATH = ROOT / "models" / "05_random_forest_results.json"
RF_MODEL_PATH = ROOT / "models" / "05_final_random_forest_model.joblib"
XGB_RESULTS_PATH = ROOT / "models" / "06_xgboost_results.json"
XGB_MODEL_PATH = ROOT / "models" / "06_final_xgboost_model.json"
STACK_RESULTS_PATH = ROOT / "models" / "07_stacking_results.json"
LR_MODEL_PATH = ROOT / "models" / "07_final_linear_regression_model.joblib"
META_MODEL_PATH = ROOT / "models" / "07_stacking_meta_model.joblib"
RESULTS_08_PATH = ROOT / "models" / "08_final_test_results.json"
HYBRID_RESULTS_PATH = ROOT / "models" / "09_hybrid_rf_arima_results.json"

ENSEMBLES = ["Random Forest", "XGBoost", "Stacking"]
TEST_MODELS = ["Persistence", "Linear Regression", "Random Forest", "XGBoost", "Stacking"]
MODEL_TYPE = {"Random Forest": "Bagging", "XGBoost": "Boosting", "Stacking": "Stacking (RF + XGBoost + LR)",
              "Linear Regression": "Single model (base model of the stack)",
              "Persistence": "Baseline (next week = this week)"}

st.set_page_config(page_title="Sri Lanka Dengue Dashboard", layout="wide")

# chart colours (same colours as notebooks 04-08)
MODEL_COLORS = {"Persistence": "#898781", "Linear Regression": "#8f6bb3",
                "Random Forest": "#2a78d6", "XGBoost": "#1baf7a", "Stacking": "#d6457f"}
RISK_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}

dark_mode = st.context.theme.type == "dark"
INK = "#f0efec" if dark_mode else "#0b0b0b"
MUTED_INK = "#c3c2b7" if dark_mode else "#52514e"
GRID = "#2c2c2a" if dark_mode else "#e1e0d9"


# ---------------------------------------------------------------- loading data and models

@st.cache_data
def load_weekly_data():
    return pd.read_csv(DATA_PATH, parse_dates=["start_date", "end_date"])


@st.cache_data
def load_thresholds():
    return pd.read_csv(THRESH_PATH, index_col="district")


@st.cache_resource
def load_models():
    """The final models, all trained on target weeks 2007-2023."""
    xgb = XGBRegressor()
    xgb.load_model(XGB_MODEL_PATH)       # notebook 06 (boosting)
    rf = joblib.load(RF_MODEL_PATH)      # notebook 05 (bagging)
    lr = joblib.load(LR_MODEL_PATH)      # notebook 07 (base model of the stack)
    meta = joblib.load(META_MODEL_PATH)  # notebook 07 (stacking meta-model)
    return {"Random Forest": rf, "XGBoost": xgb, "Linear Regression": lr, "Stacking meta-model": meta}


rf_results = json.loads(RF_RESULTS_PATH.read_text())
stack_results = json.loads(STACK_RESULTS_PATH.read_text())
xgb_results = json.loads(XGB_RESULTS_PATH.read_text())
results_08 = json.loads(RESULTS_08_PATH.read_text()) if RESULTS_08_PATH.exists() else None
hybrid_results = json.loads(HYBRID_RESULTS_PATH.read_text()) if HYBRID_RESULTS_PATH.exists() else None
FEATURES = rf_results["features"]
TARGET = rf_results["target"]
BASE_MODELS = stack_results["base_models"]   # column order the meta-model was trained with

weekly = load_weekly_data()
thresholds = load_thresholds()
models = load_models()


def base_predictions(X):
    """The three stacking base predictions, in the column order of notebook 07."""
    return pd.DataFrame({
        "Random Forest": models["Random Forest"].predict(X[FEATURES]),
        "XGBoost": np.clip(models["XGBoost"].predict(X[FEATURES]), 0, None),
        "Linear Regression": models["Linear Regression"].predict(X[FEATURES]),
    }, index=X.index)[BASE_MODELS]


def predict_with(model_name, X):
    """Predictions of one ensemble model; negative values are clipped to 0 (as in notebooks 05-08)."""
    if model_name == "Stacking":
        return np.clip(models["Stacking meta-model"].predict(base_predictions(X)), 0, None)
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
    """Low / Medium / High from the district's 75th and 90th percentile of weekly cases, 2007-2023 (notebook 03)."""
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
    """Final test (target weeks 2024-2026), the same rows as notebooks 05-08."""
    test = pd.read_csv(TEST_PATH, parse_dates=["start_date", "target_date"])
    test = test[~(test["is_imputed"] | test["target_imputed"])].reset_index(drop=True)  # leave out interpolated weeks
    test["actual"] = test[TARGET]
    test["Persistence"] = test["current_week_cases"].astype(float)
    test["Linear Regression"] = models["Linear Regression"].predict(test[FEATURES])  # not clipped (notebook 07)
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


# ---------------------------------------------------------------- chart helpers

def style_chart(fig, y_title, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=80, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      barcornerradius=4, yaxis_title=y_title)
    fig.update_yaxes(rangemode="tozero", gridcolor=GRID, zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def add_threshold_lines(fig, district):
    for level, column in [("Medium", "medium_above"), ("High", "high_above")]:
        value = thresholds.loc[district, column]
        fig.add_hline(y=value, line_dash="dash", line_width=1.5, line_color=RISK_COLORS[level],
                      annotation_text=f"{level} risk above {value:.0f}", annotation_position="top left",
                      annotation_font_color=MUTED_INK)


# ---------------------------------------------------------------- sidebar

districts = sorted(weekly["district"].unique())
latest_date = weekly["start_date"].max()
forecast_date = latest_date + pd.Timedelta(weeks=1)

st.sidebar.title("Dengue Dashboard")
st.sidebar.caption("Kularathna G.G.A.S | 224110P | Assignment 1")
district = st.sidebar.selectbox("Reporting area", districts, index=districts.index("Colombo"))
st.sidebar.markdown(f"**Data until:** week starting {latest_date:%d %b %Y}")
st.sidebar.markdown("**Models:** Random Forest (bagging), XGBoost (boosting), Stacking. "
                    "Trained on 2007-2023, tested on 2024-2026.")
st.sidebar.caption("Data: Weekly Epidemiological Reports, Epidemiology Unit, Ministry of Health, Sri Lanka "
                   "(compiled in the denguedatahub repository).")

st.title("Sri Lanka Dengue Forecasting: Bagging vs Boosting vs Stacking")
st.markdown("**Kularathna G.G.A.S | 224110P | Assignment 1**")
st.caption("Each model predicts **next week's dengue cases** for each of the 26 reporting areas from the last few "
           "weeks of cases.")

tab_compare, tab_predict, tab_data = st.tabs(["Compare models", "Predict next week", "Case history"])


# ---------------------------------------------------------------- tab 1: compare models

with tab_compare:
    test = test_period_predictions()
    test_scores = score_table(test)
    ens = test_scores.loc[ENSEMBLES]
    best_overall = test_scores["MAE"].idxmin()
    best_ensemble = ens["MAE"].idxmin()

    weights = stack_results["meta_weights"]
    by_year = pd.DataFrame({
        str(year): {name: mean_absolute_error(part["actual"], part[name]) for name in TEST_MODELS}
        for year, part in test.groupby(test["target_date"].dt.year)}).T
    rf_wins = ens.loc["Random Forest", "MAE"] < ens.loc["XGBoost", "MAE"]
    bag_winner, bag_loser = ("Random Forest (bagging)", "XGBoost (boosting)") if rf_wins else \
        ("XGBoost (boosting)", "Random Forest (bagging)")

    st.subheader("Which ensemble predicts dengue cases best?")
    st.success(f"**Answer: {best_ensemble} is the best ensemble** (lowest error on unseen 2024-2026 data), "
               f"and **{bag_winner} beats {bag_loser}**.")
    st.markdown(f"All models were trained on 2007-2023 and tested on **{len(test):,} district-weeks from 2024-2026** "
                "that they never saw. **MAE** = average prediction error in cases per week: **lower is better**.")

    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Best ensemble", best_ensemble, f"MAE {ens.loc[best_ensemble, 'MAE']:.2f}", delta_color="off")
    col2.metric("Bagging vs boosting",
                f"RF {ens.loc['Random Forest', 'MAE']:.2f} vs XGB {ens.loc['XGBoost', 'MAE']:.2f}",
                f"{bag_winner.split(' (')[0]} wins", delta_color="off")
    col3.metric("Best single model (reference)", best_overall, f"MAE {test_scores.loc[best_overall, 'MAE']:.2f}",
                delta_color="off")

    # ------------------------------------------------ 1. results
    st.divider()
    st.markdown("### 1. Results: comparing the models")
    left, right = st.columns([3, 2])
    with left:
        order = test_scores.sort_values("MAE").index
        fig = go.Figure(go.Bar(x=list(order), y=test_scores.loc[order, "MAE"],
                               marker_color=[MODEL_COLORS[name] for name in order],
                               text=[f"{v:.2f}" for v in test_scores.loc[order, "MAE"]], textposition="outside",
                               textfont_color=INK, hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
        fig.add_hline(y=test_scores.loc["Persistence", "MAE"], line_dash="dash", line_width=1, line_color=MUTED_INK,
                      annotation_text="baseline", annotation_position="top right", annotation_font_color=MUTED_INK)
        fig.update_layout(title="Prediction error on 2024-2026 (lower = better)", hovermode="closest",
                          showlegend=False)
        style_chart(fig, "MAE, cases per week")
        fig.update_yaxes(range=[0, test_scores["MAE"].max() * 1.15])
        st.plotly_chart(fig, width="stretch")
    with right:
        table = test_scores.loc[order, ["MAE", "RMSE", "MAE vs persistence (%)"]]
        table.insert(0, "type", [MODEL_TYPE[name] for name in order])
        st.markdown("**Scores table** (best value highlighted)")
        st.dataframe(table.style.format(precision=2)
                     .highlight_min(subset=["MAE", "RMSE", "MAE vs persistence (%)"], color="#1baf7a55"),
                     width="stretch")
        st.caption("RMSE = like MAE, but big misses count more. Negative % = better than the baseline "
                   "(persistence: next week = this week).")

    with st.expander("More ensemble variants tested in the notebooks (plain bagging, GBM, CatBoost, voting) "
                     "and training time"):
        other = xgb_results.get("other_boosting", {})
        variants = {
            "Bagging (all features per split)": (rf_results.get("bagging", {}).get("test_scores"),
                                                 rf_results.get("bagging", {}).get("fit_seconds_final")),
            "Random Forest (m = 5 features per split)": (rf_results["test_scores"]["Random Forest"],
                                                          rf_results.get("fit_seconds_final")),
            "XGBoost": (xgb_results["test_scores"]["XGBoost"], xgb_results.get("fit_seconds_final")),
            "GBM (scikit-learn)": (other.get("GBM", {}).get("test_scores"),
                                   other.get("GBM", {}).get("fit_seconds_final")),
            "CatBoost": (other.get("CatBoost", {}).get("test_scores"),
                         other.get("CatBoost", {}).get("fit_seconds_final")),
            "Stacking (learned weights)": (stack_results["test_scores"]["Stacking"], None),
            "Voting (equal weights)": (stack_results["test_scores"].get("Voting"), None),
        }
        variant_table = pd.DataFrame({name: {"MAE": s["MAE"], "RMSE": s["RMSE"], "training time (s)": sec}
                                      for name, (s, sec) in variants.items() if s}).T.sort_values("MAE")
        st.dataframe(variant_table.style.format(precision=2, na_rep="-"), width="stretch")
        st.markdown("- **Random Forest vs plain bagging:** choosing from 5 random features at each split makes the "
                    "trees less alike, so their average is slightly better, and training is about twice as fast.\n"
                    "- **GBM vs XGBoost:** same settings, but GBM tries exact split points. XGBoost rounds every "
                    "feature into 256 bins, which is very coarse in the rare 1,000+ case range, so it overshoots "
                    "outbreak peaks. With exact splits XGBoost behaves like GBM.\n"
                    "- **Voting vs stacking:** equal weights tie on MAE but lose on RMSE, because XGBoost's big "
                    "overestimates always get a third of the vote. Stacking learned to almost ignore them.")
        st.caption("Training time of the final model on the computer that ran the notebooks.")

    # ------------------------------------------------ 2. why
    st.divider()
    st.markdown("### 2. Why the models perform this way")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Why {bag_winner.split(' (')[0]} beat {bag_loser.split(' (')[0]}**")
        st.markdown(f"XGBoost over-predicted the peaks of the 2026 outbreak. Its 2026 error was "
                    f"**{by_year.loc['2026', 'XGBoost']:.2f}**, against **{by_year.loc['2026', 'Random Forest']:.2f}** "
                    "for the Random Forest. Averaging many trees (bagging) made the Random Forest more stable.")
    with col2:
        st.markdown("**Why the trees struggled in 2026**")
        st.markdown("In the normal years 2024 and 2025 the tree models were the best. In the big 2026 outbreak both "
                    "did worse than the baseline: the **Random Forest is limited by the range of its training "
                    "targets**, while **XGBoost sometimes amplified sharp increases and overshot** (mostly because of its "
                    "256-bin histogram splits; GBM with exact splits overshot much less). Linear Regression "
                    "handled this outbreak better.")
    with col3:
        st.markdown(f"**Why {best_ensemble} won**")
        st.markdown(f"The meta-model weights were **RF {weights['Random Forest']:.2f}**, "
                    f"**XGBoost {weights['XGBoost']:.2f}** and **Linear Regression {weights['Linear Regression']:.2f}**. "
                    "RF and XGBoost made almost the same mistakes in the walk-forward years (2016-2023), so the stack mostly followed Linear Regression, "
                    "which can follow a rising outbreak. That is also why Linear Regression alone scored slightly "
                    "better.")

    st.markdown("**Error by year:** the trees win the normal years, the 2026 outbreak decides the ranking")
    st.dataframe(by_year[TEST_MODELS].style.format(precision=2).highlight_min(axis=1, color="#1baf7a55"),
                 width="stretch")
    st.caption("Test MAE per year; the best model in each year is highlighted. 2026 has data until August.")

    # ------------------------------------------------ 3. actual vs predicted
    st.divider()
    st.markdown(f"### 3. See it for yourself: actual vs predicted cases in {district}")
    part = test[test["district"] == district]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=part["target_date"], y=part["actual"], mode="lines", name="Actual",
                             line=dict(color=INK, width=2.5), hovertemplate="actual: %{y:,}<extra></extra>"))
    for name in TEST_MODELS:
        fig.add_trace(go.Scatter(x=part["target_date"], y=part[name], mode="lines", name=name,
                                 line=dict(color=MODEL_COLORS[name], width=1.8),
                                 visible=True if name in ENSEMBLES else "legendonly",
                                 hovertemplate=f"{name}: %{{y:,.0f}}<extra></extra>"))
    fig.update_layout(title=f"{district}, 2024-2026: black = actual cases, coloured = model predictions")
    st.plotly_chart(style_chart(fig, "cases per week", height=420), width="stretch")
    st.caption("Change the reporting area in the sidebar. Click a legend entry to show or hide a model. "
               "Watch the 2026 peak: XGBoost overshoots it most.")

    # ------------------------------------------------ 4. feature importance
    if results_08:
        st.divider()
        st.markdown("### 4. What the models rely on: the most recent weeks' cases")
        importance = pd.DataFrame(results_08["feature_importance"]).T.rename(columns={
            f"{name} (test MAE increase)": name for name in ENSEMBLES})[ENSEMBLES]
        order = importance.sort_values("Random Forest").index
        fig = go.Figure()
        for name in ENSEMBLES:
            fig.add_trace(go.Bar(y=order, x=importance.loc[order, name], orientation="h", name=name,
                                 marker_color=MODEL_COLORS[name], hovertemplate=f"{name}: %{{x:.2f}}<extra></extra>"))
        fig.update_layout(title="Feature importance: how much the error grows when a feature is shuffled "
                                "(longer bar = more important)", barmode="group", hovermode="closest")
        style_chart(fig, None, height=460)
        fig.update_xaxes(title="increase in test MAE (cases)", showgrid=True, gridcolor=GRID)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("All models rely mostly on this week's cases and the last few weeks. Stacking uses the previous "
                   "week much more, through Linear Regression.")

    if hybrid_results:
        st.divider()
        st.markdown("### 5. Hybridization: Random Forest + ARIMA")
        hyb = pd.DataFrame(hybrid_results["test_scores"]).T[["MAE", "RMSE"]]
        col1, col2 = st.columns([2, 3])
        col1.dataframe(hyb.round(2), width="stretch")
        col2.markdown(
            "ARIMA predicts the Random Forest's **next error** from its recent errors, and the hybrid adds that "
            f"correction. It made the forecasts **worse** (MAE **{hyb.loc['RF + ARIMA', 'MAE']:.2f}** vs "
            f"**{hyb.loc['Random Forest', 'MAE']:.2f}** for the Random Forest alone). The Random Forest's errors are "
            f"almost unrelated from week to week (median lag-1 autocorrelation "
            f"**{hybrid_results['lag1_autocorrelation_2023_median']:.2f}**), so ARIMA has no pattern to learn and "
            "its corrections mostly add noise. See notebook 09.")


# ---------------------------------------------------------------- tab 2: predict next week

with tab_predict:
    st.subheader("Predict next week's dengue cases and risk")
    model_name = st.radio("Choose an ensemble model", ENSEMBLES, horizontal=True,
                          format_func=lambda name: f"{name} ({MODEL_TYPE[name].split(' (')[0].lower()})"
                          if name != "Stacking" else "Stacking")
    forecast = predict_next_week(model_name)
    row = forecast.loc[district]

    st.markdown(f"#### {district}: forecast for the week starting {row['target_date']:%d %b %Y}")

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Reported, week of {row['start_date']:%d %b}", f"{row['cases']:,}")
    change = row["predicted_cases"] - row["cases"]
    col2.metric(f"Predicted ({model_name})", f"{row['predicted_cases']:,}",
                delta=f"{change:+,} vs latest week" if change != 0 else "same as latest week",
                delta_color="inverse" if change != 0 else "off")
    with col3:
        st.markdown("**Risk level**")
        {"Low": st.success, "Medium": st.warning, "High": st.error}[row["risk"]](f"**{row['risk']} risk**")

    st.markdown("**The three ensembles compared for " + district + ":** "
                + " | ".join(f"{name} **{row[name]:,}**" for name in ENSEMBLES) + " cases")

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
    fig.update_layout(title=f"{district}: last 12 weeks and next-week prediction")
    st.plotly_chart(style_chart(fig, "cases per week"), width="stretch")

    medium_above = thresholds.loc[district, "medium_above"]
    high_above = thresholds.loc[district, "high_above"]
    st.caption(f"Risk for {district}: Low up to {medium_above:.0f} cases, Medium up to {high_above:.0f}, High above "
               f"{high_above:.0f} (the area's own 75th and 90th percentile, 2007-2023). These are statistical levels, "
               "not official outbreak alerts. The data ends in August 2026, so this is a forecast for that time, "
               "not for today.")

    st.markdown(f"#### Risk level for all 26 reporting areas ({model_name})")
    counts = forecast["risk"].value_counts()
    col1, col2, col3 = st.columns(3)
    for col, level in zip([col1, col2, col3], ["High", "Medium", "Low"]):
        col.metric(f"{level} risk", counts.get(level, 0))
    table = forecast.reset_index().sort_values("predicted_cases", ascending=False)
    st.dataframe(
        table[["district", "cases", *ENSEMBLES, "risk"]].rename(columns={
            "district": "reporting area", "cases": f"reported (week of {latest_date:%d %b})",
            "risk": f"risk ({model_name})"}),
        hide_index=True, width="stretch", height=950)


# ---------------------------------------------------------------- tab 3: data

with tab_data:
    st.subheader(f"Dengue Cases in Sri Lanka, {int(weekly['year'].min())}-{int(weekly['year'].max())}")
    st.caption("Weekly dengue cases reported by the Epidemiology Unit, Ministry of Health, for 26 reporting areas.")
    national = weekly.groupby("start_date", as_index=False)["cases"].sum()
    fig = go.Figure(go.Scatter(x=national["start_date"], y=national["cases"], mode="lines", name="National cases",
                               line=dict(color=INK, width=2), hovertemplate="%{y:,} cases<extra></extra>"))
    fig.update_layout(title="Weekly dengue cases in Sri Lanka, all 26 reporting areas")
    st.plotly_chart(style_chart(fig, "cases per week"), width="stretch")

    first_year, last_year = int(weekly["year"].min()), int(weekly["year"].max())
    start_year, end_year = st.slider("Years", first_year, last_year, (last_year - 4, last_year))
    history = weekly[(weekly["district"] == district) & weekly["year"].between(start_year, end_year)]
    fig = go.Figure(go.Scatter(x=history["start_date"], y=history["cases"], mode="lines", name="Weekly cases",
                               line=dict(color=INK, width=2), hovertemplate="%{y:,} cases<extra></extra>"))
    add_threshold_lines(fig, district)
    fig.update_layout(title=f"{district}, {start_year}-{end_year}")
    st.plotly_chart(style_chart(fig, "cases per week", height=420), width="stretch")
