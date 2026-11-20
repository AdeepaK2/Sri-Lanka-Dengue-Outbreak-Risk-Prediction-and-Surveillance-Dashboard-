"""
Sri Lanka Dengue Surveillance Dashboard

Run from the project folder:
    streamlit run app/dashboard.py

Uses the files made by the notebooks:
    dataset/processed/02_cleaned_weekly_data.csv   weekly cases per district (notebook 02)
    dataset/processed/03_risk_thresholds.csv       district risk thresholds (notebook 03)
    dataset/processed/test.csv                     test weeks 2024-2026 (notebook 03)
    models/05_selected_model.json                  validation scores of all models (notebook 05)
    models/06_final_xgboost_model.json             final trained XGBoost model (notebook 06)
"""
import json
from pathlib import Path

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
CONFIG_PATH = ROOT / "models" / "05_selected_model.json"
MODEL_PATH = ROOT / "models" / "06_final_xgboost_model.json"

st.set_page_config(page_title="Sri Lanka Dengue Dashboard", layout="wide")

# chart colours (models use the same colours as notebooks 04-06)
MODEL_COLORS = {"Persistence": "#898781", "Decision Tree": "#eb6834", "Random Forest": "#2a78d6", "XGBoost": "#1baf7a"}
BAR_COLOR = "#2a78d6"
RISK_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}

dark_mode = st.context.theme.type == "dark"
INK = "#f0efec" if dark_mode else "#0b0b0b"         # actual case counts and text
MUTED_INK = "#c3c2b7" if dark_mode else "#52514e"
GRID = "#2c2c2a" if dark_mode else "#e1e0d9"

DISCLAIMER = ("Risk levels are **relative statistical levels** for each district, based on its own past weekly "
              "case counts. They are **not official outbreak alerts** from the Ministry of Health.")


# ---------------------------------------------------------------- loading data and model

@st.cache_data
def load_weekly_data():
    return pd.read_csv(DATA_PATH, parse_dates=["start_date", "end_date"])


@st.cache_data
def load_thresholds():
    return pd.read_csv(THRESH_PATH, index_col="district")


@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


config = json.loads(CONFIG_PATH.read_text())
FEATURES = config["features"]

weekly = load_weekly_data()
thresholds = load_thresholds()
model = load_model()


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


def predict_next_week():
    """Predict next week's cases for every district from its latest reported week."""
    latest = add_features(weekly).groupby("district").tail(1).copy()
    predicted = np.clip(model.predict(latest[FEATURES]), 0, None)  # no negative case counts
    latest["predicted_cases"] = np.round(predicted).astype(int)
    latest["target_date"] = latest["start_date"] + pd.Timedelta(weeks=1)
    latest["risk"] = [risk_level(c, d) for c, d in zip(latest["predicted_cases"], latest["district"])]
    return latest.set_index("district")


@st.cache_data
def test_period_scores():
    """Final test (2024-2026): XGBoost vs persistence, the same as notebook 06."""
    test = pd.read_csv(TEST_PATH, parse_dates=["start_date", "target_date"])
    test = test[~(test["is_imputed"] | test["target_imputed"])]  # leave out interpolated weeks
    actual = test[config["target"]]
    predictions = {
        "Persistence": test["current_week_cases"],
        "XGBoost": np.clip(load_model().predict(test[FEATURES]), 0, None),
    }
    return pd.DataFrame({name: {"MAE": mean_absolute_error(actual, pred), "RMSE": root_mean_squared_error(actual, pred)}
                         for name, pred in predictions.items()}).T


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

st.sidebar.title("Dengue Dashboard")
district = st.sidebar.selectbox("District", districts, index=districts.index("Colombo"))
st.sidebar.caption("The district is used in the *Historical trends* and *Prediction & risk* tabs.")
st.sidebar.divider()
st.sidebar.markdown(f"**Latest data:** week starting {latest_date:%d %b %Y} ({latest_year}, week {latest_week_no})")
st.sidebar.markdown(f"**Model:** {config['selected_model']}, trained on {config['validation_year']} and earlier")
st.sidebar.caption("Data: Weekly Epidemiological Reports, Epidemiology Unit, Ministry of Health, Sri Lanka, "
                   "compiled in the denguedatahub repository (Dr. Thiyanga S. Talagala).")

st.title("Sri Lanka Dengue Surveillance Dashboard")

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
    col3.metric("Most cases in the latest week", top_district["district"], delta=f"{top_district['cases']:,} cases",
                delta_color="off")

    fig = go.Figure(go.Scatter(x=national["start_date"], y=national["cases"], mode="lines", name="National cases",
                               line=dict(color=INK, width=2),
                               hovertemplate="%{y:,} cases<extra></extra>"))
    fig.update_layout(title="Weekly dengue cases, all 26 districts")
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
        fig.update_layout(title=f"Cases by district, week starting {latest_date:%d %b %Y}", hovermode="closest")
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
    st.caption("Dashed lines are the district's risk thresholds. " + DISCLAIMER)

    with st.expander(f"Show data: {district} weekly cases"):
        st.dataframe(history[["year", "week", "start_date", "cases", "is_imputed"]]
                     .sort_values("start_date", ascending=False)
                     .rename(columns={"start_date": "week starting", "is_imputed": "estimated"}),
                     hide_index=True, width="stretch")


# ---------------------------------------------------------------- tab 3: next-week prediction and risk

with tab_prediction:
    forecast = predict_next_week()
    row = forecast.loc[district]

    st.subheader(f"Next-week prediction for {district}")
    st.caption(f"Predicted with the final XGBoost model from the latest reported week "
               f"(starting {row['start_date']:%d %b %Y}).")

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Reported cases, week starting {row['start_date']:%d %b}", f"{row['cases']:,}")
    col2.metric(f"Predicted cases, week starting {row['target_date']:%d %b}", f"{row['predicted_cases']:,}",
                delta=f"{row['predicted_cases'] - row['cases']:+,} vs latest week", delta_color="inverse")
    with col3:
        st.markdown("**Risk level for next week**")
        show_risk(row["risk"])

    recent = weekly[weekly["district"] == district].tail(12)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent["start_date"], y=recent["cases"], mode="lines+markers", name="Reported cases",
                             line=dict(color=INK, width=2), marker=dict(size=8),
                             hovertemplate="reported: %{y:,}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[row["start_date"], row["target_date"]], y=[row["cases"], row["predicted_cases"]],
                             mode="lines+markers", name="XGBoost prediction",
                             line=dict(color=MODEL_COLORS["XGBoost"], width=2, dash="dot"),
                             marker=dict(size=[0, 11], color=MODEL_COLORS["XGBoost"]),
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
    st.subheader("Risk classification for all districts")

    counts = forecast["risk"].value_counts()
    col1, col2, col3 = st.columns(3)
    for col, level in zip([col1, col2, col3], ["High", "Medium", "Low"]):
        n = counts.get(level, 0)
        col.metric(f"{level} risk", f"{n} district" if n == 1 else f"{n} districts")

    table = forecast.reset_index().join(thresholds, on="district")
    table = table.sort_values(["predicted_cases"], ascending=False)
    st.dataframe(
        table[["district", "cases", "predicted_cases", "risk", "medium_above", "high_above"]].rename(columns={
            "cases": f"reported (week of {latest_date:%d %b})",
            "predicted_cases": "predicted next week",
            "medium_above": "Medium above",
            "high_above": "High above",
        }),
        hide_index=True, width="stretch", height=950,
        column_config={"Medium above": st.column_config.NumberColumn(format="%.0f"),
                       "High above": st.column_config.NumberColumn(format="%.0f")},
    )


# ---------------------------------------------------------------- tab 4: model comparison

with tab_models:
    st.subheader("Decision Tree vs Random Forest vs XGBoost")
    st.markdown("""
- **Persistence (baseline):** next week's cases = this week's cases. A model is only useful if it beats this.
- **Decision Tree:** one tree of yes/no questions about the features.
- **Random Forest (bagging):** many trees trained on random samples of the data; their predictions are averaged.
- **XGBoost (boosting):** trees built one after another, each one correcting the errors of the trees before it.
""")

    scores = pd.DataFrame(config["validation_scores"]).T
    scores = scores.loc[["Persistence", "Decision Tree", "Random Forest", "XGBoost"]]
    scores["MAE vs persistence (%)"] = 100 * (scores["MAE"] / scores.loc["Persistence", "MAE"] - 1)

    st.markdown(f"**Validation results ({config['validation_year']})**: trained on 2007-{config['validation_year'] - 1}, "
                f"tested on every district-week of {config['validation_year']}. Lower is better.")

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
               "RMSE = like MAE, but large errors count more.")

    st.markdown(f"**Selected model: {config['selected_model']}** (lowest validation MAE). "
                "All three tree models are close to the persistence baseline: dengue cases change slowly "
                "from week to week, so this week's count is already a strong guide to next week.")

    st.divider()
    st.markdown("**Final test (2024-2026)**: XGBoost retrained on 2007-2023 and tested once on weeks it never saw.")
    test_scores = test_period_scores()
    test_scores["MAE vs persistence (%)"] = 100 * (test_scores["MAE"] / test_scores.loc["Persistence", "MAE"] - 1)
    st.dataframe(test_scores.round(2), width="stretch")

    if test_scores.loc["XGBoost", "MAE"] < test_scores.loc["Persistence", "MAE"]:
        st.markdown("On the test weeks XGBoost has a lower MAE than the persistence baseline.")
    else:
        st.warning("On the 2024-2026 test weeks XGBoost did **not** beat the persistence baseline "
                   "(higher MAE and RMSE). Treat its predictions as a rough guide only.")
