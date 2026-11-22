# Sri-Lanka-Dengue-Outbreak-Risk-Prediction-and-Surveillance-Dashboard-

Ensemble machine learning (bagging (Random Forest), boosting (XGBoost) and stacking) for one-week-ahead forecasting of dengue cases in Sri Lanka, with a Streamlit surveillance dashboard.

## 1. Problem statement and objectives

Dengue is a major public-health problem in Sri Lanka, with large outbreaks such as 2017 and 2026. Weekly case counts are published for each reporting area, and an early estimate of next week's cases can help with planning.

**Problem:** predict the number of dengue cases **next week** in each of the 26 reporting areas from the weekly surveillance data. This is a regression problem with a time-series structure.

**Objectives**
1. Clean and validate the Sri Lankan weekly dengue dataset and build leakage-free features.
2. Compare **bagging (Random Forest)**, **boosting (XGBoost)** and **stacking** (Random Forest + XGBoost + Linear Regression) against simple baselines (persistence, a single Decision Tree and Multiple Linear Regression) using a strict chronological evaluation.
3. Find out *why* the models succeed or fail, especially during outbreaks.
4. Present the data, forecasts, relative risk levels and model comparison in a **Streamlit dashboard**.

## 2. Dataset Source and Acknowledgment

The Sri Lankan weekly dengue surveillance dataset used in this project was obtained from the **denguedatahub** repository developed and maintained by **Dr. Thiyanga S. Talagala**.

The dataset compiles district-level dengue surveillance information originally published in the Weekly Epidemiological Reports of the Epidemiology Unit, Ministry of Health, Sri Lanka.

We acknowledge Dr. Talagala for collecting, organizing, and making these data publicly accessible for research and analysis.

**Dataset repository:** https://github.com/thiyangt/denguedatahub

**Original data provider:** Epidemiology Unit, Ministry of Health, Sri Lanka.

## 3. Dataset characteristics and cleaning

**Raw file:** `dataset/raw/srilanka_weekly_data.csv` (never modified). It has 26,753 rows with year, week, start date, end date, area and case count, covering 1,029 weeks from 2006-12-23 to 2026-09-13.

- **26 reporting areas:** the 25 administrative districts plus **Kalmunai**, which the Epidemiology Unit reports as a separate health area (spelt `Kalmune` in the data).
- Counts are very skewed: about 14% of rows are zero, and outbreak peaks reach 2,631 cases in one week (Gampaha, July 2017). The very high weeks are real outbreaks and are kept.

**Cleaning decisions** (notebooks 01-02)

| Issue | Decision |
|---|---|
| Mixed date formats (2026 weeks 21-37) | parsed both formats |
| Weeks not exactly 7 days apart (2009, 2026) | all weeks re-spaced 7 days apart, which moves some dates by 1-2 days |
| Missing Puttalam row (2026 week 7) | filled with the mean of weeks 6 and 8, flagged `is_imputed` |
| Counts attached to the wrong areas (2026 weeks 14 and 24) and a collapsed week (2022 week 37) | replaced by linear interpolation, flagged `is_imputed` |
| 2026 week 37 (wrong areas, last week of the data) | dropped, so the cleaned data ends with the week starting 2026-08-29 |
| Four pairs of identical consecutive weeks (2012, 2021, 2025, 2026) | kept and flagged `dup_week`, because it is unknown which copy is real |
| Sharp drops in late August 2026 (possibly late reports) | kept as reported; listed as a limitation |

The cleaned data (`02_cleaned_weekly_data.csv`) has 26,728 rows, of which 79 are imputed and 208 are flagged as duplicated weeks.

## 4. Feature engineering (notebook 03)

There is one row per reporting area and week. Every feature uses only the **current week and earlier weeks**:

| Feature | Meaning |
|---|---|
| `current_week_cases` | cases this week |
| `previous_week_cases`, `previous_2_weeks_cases`, `previous_3_weeks_cases` | lags 1-3 |
| `seasonal_cases_previous_52_weeks` | same week one year earlier (lag 52) |
| `moving_average_4_weeks`, `moving_average_8_weeks` | mean of the last 4 / 8 weeks, including the current week |
| `weekly_case_change` | current week minus previous week |
| `seasonal_week_sin`, `seasonal_week_cos` | position in the year, encoded as a circle |
| `district_code` | reporting area, coded in alphabetical order |

- **Target:** `cases_next`, the cases in the following week. Its date is checked to be exactly 7 days later.
- **Leakage rule:** rows whose current week (`is_imputed`) or target week (`target_imputed`) was interpolated are removed before training and evaluation. An interpolated value is computed from the following week, which is the answer the model is trying to predict.
- **Risk thresholds:** the 75th and 90th percentile of each area's weekly cases, computed from weeks up to 2023-12-31 only (`03_risk_thresholds.csv`).

## 5. Training, validation and testing periods

| Set | Target weeks | Rows (after the leakage rule) | Used for |
|---|---|---|---|
| Training | 2007-12-29 to 2022-12-31 | 20,332 | fitting the models while choosing settings |
| Validation | 2023-01-07 to 2023-12-30 | 1,352 | comparing settings and models (notebooks 04-07) |
| Final training | 2007-12-29 to 2023-12-30 | 21,684 | refitting the chosen settings (notebooks 05-07) |
| **Test** | 2024-01-06 to 2026-08-29 | **3,508** | final evaluation only; never used for any decision |

The split is chronological: models always learn from the past and are evaluated on the future. For stacking (notebook 07), the base models' predictions for 2016-2023 are made walk-forward: each year is predicted by base models trained on earlier years only.

## 6. Models and hyperparameters

All settings were chosen on the **2023 validation year only** (notebook 04: Decision Tree, notebook 05: Random Forest, notebook 06: XGBoost, notebook 07: stacking meta-model), from a few simple candidate settings per model. `random_state = 42` is used everywhere.

| Model | Type | Settings | Chosen from |
|---|---|---|---|
| Persistence | baseline | next week = this week | - |
| Decision Tree | single tree | `max_depth=5` | no limit, 10, 5 |
| Linear Regression | baseline / stacking base model | `LinearRegression()` (multiple linear regression on the 11 features) | - (no settings) |
| Random Forest | bagging | `n_estimators=200`, `min_samples_leaf=10` | default (100 trees); 200 trees with `min_samples_leaf=10`; 200 trees with `max_depth=10` |
| XGBoost | boosting | `n_estimators=400`, `max_depth=2`, `learning_rate=0.05` | (100, 6, 0.3), (200, 3, 0.1), (400, 2, 0.05) |
| Stacking | stacking | RF + XGB + LR → meta-model, chosen on 2023: `LinearRegression(positive=True)` | Random Forest meta-model (100 trees); Linear Regression meta-model (`positive=True`) |

XGBoost and stacking predictions are clipped at 0. A Random Forest cannot predict below 0. The Linear Regression is used without clipping.

## 7. Bagging vs boosting vs stacking

| | Bagging | Boosting | Stacking |
|---|---|---|---|
| Base models | many models of the **same** type (here: trees) | many models of the **same** type (here: small trees) | a few models of **different** types |
| How they are trained | **independently, in parallel**, each on a bootstrap sample | **sequentially**, each one fitted to the errors of the models before it | base models first, then a **meta-model** trained on their predictions |
| How predictions are combined | **average** (regression) or majority vote (classification) | **weighted sum** of all models | **learned** by the meta-model |
| Mainly reduces | **variance** (overfitting) | **bias** (underfitting) | can reduce both, if the base models make different errors |
| Main risk | cannot predict beyond the range of past values | can overfit rare patterns; needs careful settings | **leakage** if the meta-model is trained on predictions for rows the base models were trained on; more complex |
| In this project | Random Forest (notebook 05) | XGBoost (notebook 06) | RF + XGBoost + Linear Regression (notebook 07) |

## 8. Results

### 8.1 Validation (2023, notebooks 04-07; used to choose the settings)

| Model | MAE | RMSE |
|---|---|---|
| Persistence | 18.17 | 41.82 |
| Decision Tree | 18.96 | 43.01 |
| Linear Regression | **17.93** | **40.24** |
| Random Forest | 18.14 | 42.32 |
| XGBoost | 18.10 | 41.34 |
| Stacking | 18.11 | 40.83 |

On 2023 all models except the Decision Tree are within 1.5% of persistence in MAE. XGBoost has the lowest MAE of the three ensembles, but its lead (0.04 cases per district-week over the Random Forest) is far too small to call it better.

### 8.2 Final test (2024-2026, notebooks 05-08)

Each ensemble is tested in its own notebook (05 Random Forest, 06 XGBoost, 07 Stacking), and notebook 08 compares them. All models are evaluated on the same 3,508 rows. MAE and RMSE are in cases per district-week. Negative percentages mean a smaller error than persistence.

| Model | Test MAE | Test RMSE | MAE vs persistence | RMSE vs persistence |
|---|---|---|---|---|
| Persistence | 12.63 | 29.93 | - | - |
| Linear Regression (stacking base model) | **11.90** | **26.95** | **-5.8%** | **-10.0%** |
| Random Forest | 12.53 | 33.63 | -0.8% | +12.4% |
| XGBoost | 13.22 | 39.19 | +4.6% | +30.9% |
| Stacking | 12.18 | 27.72 | -3.6% | -7.4% |

| Test year | Mean cases per district-week | Persistence MAE | Linear Regression MAE | Random Forest MAE | XGBoost MAE | Stacking MAE |
|---|---|---|---|---|---|---|
| 2024 | 37.1 | 10.45 | 9.97 | 9.53 | 9.47 | 10.24 |
| 2025 | 37.6 | 8.68 | 8.61 | 8.31 | 8.27 | 8.63 |
| 2026 (to August) | 98.9 | 22.93 | 20.66 | 24.66 | 27.82 | 21.42 |

- Among the three ensembles, **Stacking has the lowest test MAE and RMSE** and is the only one that beats persistence on both. Linear Regression on its own scored slightly better still (notebook 07).
- By MAE (comparing persistence and the three ensembles), the Random Forest is the best model in 13 of the 26 reporting areas, XGBoost in 5, Stacking in 4 and persistence in 4. Stacking is best in the three largest districts (Colombo, Gampaha, Kandy). The Random Forest beats XGBoost in 18 areas.
- The 37 district-weeks with more than 500 cases (1.1% of the test rows) account for 68% of XGBoost's squared error, 63% of the Random Forest's, 50% of persistence's and 47% of Stacking's.

**Sensitivity check (duplicated weeks).** On the 52 test rows whose target week is an exact copy of the current week, persistence scores a perfect error of 0. Without the 130 test rows that use a duplicated week, the MAE changes are: Random Forest -4.0% vs persistence, Stacking -3.6%, XGBoost +2.1%. Stacking stays clearly better on RMSE (-7.1%); both tree ensembles are still worse than persistence on RMSE. The official results above include all rows.

## 9. Key findings

1. **Random Forest generalised better than XGBoost** on the unseen years: it had lower MAE and RMSE overall and a much lower RMSE in the 2026 outbreak. XGBoost's tiny validation lead did not carry over, which shows why an untouched test set is needed.
2. **Neither tree ensemble clearly beats persistence.** Both improve on it in the moderate years 2024-2025 (MAE about 4-9% lower), but both are worse during the 2026 outbreak and have a higher overall RMSE.
3. **Why XGBoost does worse:** only 99 training weeks had more than 500 cases, and 69 of them come from the 2017 epidemic. In 2017, large weekly jumps at high levels were followed by even higher weeks. In July 2026, Colombo went from 1,264 to 1,625 cases, and XGBoost predicted about 2,265 against an actual 1,330. The jump feature (`weekly_case_change`) alone added about +759 cases. Boosting adds up the contributions of correlated features, so a pattern learned from a few rows gets amplified. The Random Forest averages at least 10 past rows per leaf and stays within past values, so it overshoots less.
4. **What the models use:** both tree ensembles depend mostly on `current_week_cases` and the 4-week moving average, with permutation importance measured on the test data. They have essentially learned a smoothed persistence rule, which is why they are so close to it.
5. **Stacking had the best test scores of the three ensembles, but not because it combined bagging and boosting.** In the years the meta-model learned from (walk-forward 2016-2023), the Random Forest and XGBoost made almost the same errors (correlation 0.98), so combining them added no information. The meta-model gave nearly all its weight to the Linear Regression base model (1.007; XGBoost 0.034; Random Forest 0), whose errors were different and which handled outbreaks better because, unlike trees, it is not limited to past values. Stacking beat persistence on both MAE (-3.6%) and RMSE (-7.4%), mainly thanks to the 2026 outbreak, but Linear Regression alone was slightly better (MAE 11.90, RMSE 26.95).

## 10. Methodology checks

| Check | Result |
|---|---|
| Data leakage through interpolated weeks | Prevented: rows with an interpolated current or target week are removed (52 training rows, 106 test rows). |
| Target in the features / target alignment | Not in the features (asserted); the target is exactly the following week (asserted in notebooks 03 and 05-08). |
| Stacking leakage | The meta-model is trained only on walk-forward base predictions: each year 2016-2023 is predicted by base models trained on earlier years only. |
| Temporal split / train-test contamination | Chronological, with `train < validation < test` asserted. Thresholds and settings use data up to 2023 only, and the test set is used once. |
| Reporting-area coding | Alphabetical codes from notebook 02, used identically by the notebooks and the dashboard. |
| Dashboard vs training features | The dashboard rebuilds the 11 features with the notebook 03 formulas; checked identical on all 3,614 test rows. Dashboard predictions are bit-identical to the evaluation predictions. |
| Reproducibility | Retraining XGBoost gives the saved model's predictions exactly. Random Forest trees are identical across runs; parallel prediction differed by about 1e-13, so the saved model predicts in a single thread (`n_jobs=1`) for exact repeatability. Notebook 07's 2023 walk-forward base predictions reproduce the validation MAE of notebooks 05 and 06 (18.139 and 18.099), and notebook 08 checks that all loaded models reproduce the test scores recorded in notebooks 05-07. |
| Metrics | MAE and RMSE are pooled over all district-weeks, so they are dominated by the large areas; per-area, per-year and outbreak-week results are also reported. |
| Duplicated weeks | Favour persistence; reported as a sensitivity check without changing the test set. |

## 11. Limitations

- The test period is a single, short period (2024 to August 2026) dominated by one large outbreak, and there is no statistical significance test of the differences.
- Settings were chosen on one validation year from a few candidates, so the validation scores are slightly optimistic.
- Only past case counts are used. There are no weather, mobility or serotype data, which are the kind of information that could signal turning points.
- The most recent 2026 weeks may be incomplete (late reports), and the duplicated weeks may hide data errors.
- The risk thresholds are fixed percentiles from 2007-2023 and do not adjust to population or reporting changes.

## 12. Research note

Random Forest (Breiman, 2001) uses bagging; XGBoost (Chen & Guestrin, 2016), LightGBM (Ke et al., 2017) and CatBoost (Prokhorenkova et al., 2018) are gradient boosting libraries, and stacking (Wolpert, 1992) combines different models through a meta-model. LightGBM's and CatBoost's main advantages are speed on very large datasets and handling of many categorical features; this dataset is small and has a single categorical feature (`district_code`), so they would be expected to behave like XGBoost, but they were **not tested**. On medium-sized tabular data like this, tree ensembles are usually as accurate as deep learning or better (Grinsztajn et al., 2022).

**References**
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD 2016*, 785-794.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30*.
- Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems 31*.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241-259.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 2022 Datasets and Benchmarks Track*.

## 13. Risk levels are not official alerts

The dashboard converts predicted case counts into **Low / Medium / High** using each reporting area's own 75th and 90th percentile of weekly cases (2007-2023). These are **relative statistical indicators**, not official outbreak alerts or public-health advice from the Ministry of Health or the Epidemiology Unit.

## 14. Project structure

```
dataset/raw/srilanka_weekly_data.csv          original data (never modified)
dataset/processed/                            outputs of notebooks 01-03 (validated, cleaned, features, train/test, thresholds)
notebooks analysis/
  01_data_understanding_validation.ipynb      checks only
  02_data_cleaning.ipynb                      cleaning decisions
  03_feature_engineering.ipynb                features, target, split, thresholds
  04_baseline_model_training.ipynb            persistence, Decision Tree, Multiple Linear Regression (validation 2023)
  05_bagging_random_forest.ipynb              BAGGING: Random Forest (tuning on 2023, final test 2024-2026)
  06_boosting_xgboost.ipynb                   BOOSTING: XGBoost (tuning on 2023, final test 2024-2026)
  07_stacking.ipynb                           STACKING: RF + XGBoost + Linear Regression -> meta-model (chosen on 2023, final test)
  08_final_comparison.ipynb                   bagging vs boosting vs stacking vs persistence on the same test rows + diagnostics
models/
  05_random_forest_results.json               Random Forest settings, validation and test scores
  05_final_random_forest_model.joblib         final Random Forest model
  06_xgboost_results.json                     XGBoost settings, validation and test scores
  06_final_xgboost_model.json                 final XGBoost model
  07_stacking_results.json                    stacking meta-model, base models, meta weights, validation and test scores
  07_final_linear_regression_model.joblib     Linear Regression base model of the stack
  07_stacking_meta_model.joblib               stacking meta-model
  08_final_test_results.json                  test scores, sensitivity check, feature importance
  08_test_predictions.csv                     every test row with actual and predicted values
app/dashboard.py                              Streamlit dashboard
```

## 15. Installation and running the dashboard

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

Run the second command from the project folder.

The dashboard has three tabs:
- **Compare models:** 2024-2026 test results for Random Forest (bagging), XGBoost (boosting) and Stacking (with persistence and Linear Regression for reference), which model did best and why, actual vs predicted for a reporting area, and feature importance.
- **Predict next week:** next-week prediction with a choice of **Random Forest, XGBoost or Stacking**, risk level, and a table for all 26 areas.
- **Data:** national weekly cases and the weekly cases of one reporting area.

The data ends with the week starting 2026-08-29, so the forecast is for the following week, not for today.

**Tested with:** Python 3.14.6, pandas 3.0.3, numpy 2.4.6, scikit-learn 1.9.0, xgboost 3.4.1, streamlit 1.58.0, plotly 6.7.0. The Random Forest, Linear Regression and meta-model files are joblib/pickle files: load them with the same scikit-learn version, and only load it from a trusted source.

## 16. Reproducing the results

Run the notebooks in order (01 to 08) from inside the `notebooks analysis/` folder, because they use relative paths such as `../dataset`:

```bash
cd "notebooks analysis"
jupyter nbconvert --to notebook --execute --inplace 05_bagging_random_forest.ipynb
jupyter nbconvert --to notebook --execute --inplace 06_boosting_xgboost.ipynb
jupyter nbconvert --to notebook --execute --inplace 07_stacking.ipynb
jupyter nbconvert --to notebook --execute --inplace 08_final_comparison.ipynb
```

- Notebooks 01-03 rewrite the files in `dataset/processed/`, and notebooks 05-08 write to `models/`.
- Notebook 07 uses the settings and saved models of notebooks 05 and 06, and notebook 08 loads the models of notebooks 05-07, so run them in order.
- All models use `random_state = 42`. Notebook 08 checks that the loaded models reproduce the test scores recorded in notebooks 05-07.

## 17. Declaration of AI assistance

An AI agentic coding assistant, **Claude** (developed by Anthropic and used through **Claude Code**), was used as a helper during the development of this project.

**How it was used**
- **Notebooks (`notebooks analysis/`):** help with writing, simplifying and explaining code and Markdown, checking the data against the original reports, and auditing the method for data leakage and correct time-based splitting. Notebooks **07** (stacking) and **08** (final comparison) and the Multiple Linear Regression section of notebook **04** were drafted with the assistant. The assistant also helped split the ensemble work into a separate bagging notebook (**05**, Random Forest) and boosting notebook (**06**, XGBoost).
- **Dashboard (`app/dashboard.py`) and this README:** drafted with the assistant.

**What remained our responsibility**
- The project idea, the dataset choice, and the decisions about scope, methods and what to report were ours.
- All code was reviewed and run by us. Every score, table and chart comes from running the notebooks on the data, not from text written by the AI.
- We checked the results and are responsible for the final content of the project, including its limitations.
