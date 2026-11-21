# Sri-Lanka-Dengue-Outbreak-Risk-Prediction-and-Surveillance-Dashboard-

Ensemble machine learning (Random Forest and XGBoost) for one-week-ahead forecasting of dengue cases in Sri Lanka, with a Streamlit surveillance dashboard.

## 1. Problem statement and objectives

Dengue is a major public-health problem in Sri Lanka, with large outbreaks such as 2017 and 2026. Weekly case counts are published for each reporting area, and an early estimate of next week's cases can help with planning.

**Problem:** predict the number of dengue cases **next week** in each of the 26 reporting areas from the weekly surveillance data. This is a regression problem with a time-series structure.

**Objectives**
1. Clean and validate the Sri Lankan weekly dengue dataset and build leakage-free features.
2. Compare **bagging (Random Forest)** and **boosting (XGBoost)** against simple baselines (persistence and a single Decision Tree) using a strict chronological evaluation.
3. Find out *why* the models succeed or fail, especially during outbreaks.
4. Test whether a **hybrid** (Random Forest + ARIMA) improves the forecasts.
5. Present the data, forecasts, relative risk levels and model comparison in a **Streamlit dashboard**.

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
| Validation | 2023-01-07 to 2023-12-30 | 1,352 | comparing settings and models (notebooks 04-05) |
| Final training | 2007-12-29 to 2023-12-30 | 21,684 | refitting the chosen settings (notebooks 06-07) |
| **Test** | 2024-01-06 to 2026-08-29 | **3,508** | final evaluation only; never used for any decision |

The split is chronological: models always learn from the past and are evaluated on the future.

## 6. Models and hyperparameters

All settings were chosen on the **2023 validation year only** (notebooks 04-05), from a few simple candidate settings per model. `random_state = 42` is used everywhere.

| Model | Type | Settings | Chosen from |
|---|---|---|---|
| Persistence | baseline | next week = this week | - |
| Decision Tree | single tree | `max_depth=5` | no limit, 10, 5 |
| Random Forest | bagging | `n_estimators=200`, `min_samples_leaf=10` | default (100 trees); 200 trees with `min_samples_leaf=10`; 200 trees with `max_depth=10` |
| XGBoost | boosting | `n_estimators=400`, `max_depth=2`, `learning_rate=0.05` | (100, 6, 0.3), (200, 3, 0.1), (400, 2, 0.05) |

XGBoost predictions are clipped at 0. A Random Forest cannot predict below 0.

## 7. Bagging vs boosting

| | Bagging (Random Forest) | Boosting (XGBoost) |
|---|---|---|
| Training | many deep trees built **independently**, each on a bootstrap sample with random feature subsets | shallow trees built **one after another**, each fitted to the errors left by the previous trees |
| Combination | **average** of the trees | **sum** of the trees, scaled by the learning rate |
| Mainly reduces | **variance** (overfitting of single trees) | **bias** (underfitting) |
| Main risk | cannot predict outside the range of training targets; large models | needs careful tuning; can fit rare patterns too strongly |

## 8. Results

### 8.1 Validation (2023, notebook 05; used to choose the settings)

| Model | MAE | RMSE |
|---|---|---|
| Persistence | 18.17 | 41.82 |
| Decision Tree | 18.96 | 43.01 |
| Random Forest | 18.14 | 42.32 |
| XGBoost | **18.10** | **41.34** |

XGBoost was selected in notebook 05, but its lead over the Random Forest (0.04 cases per district-week) is far too small to call it better.

### 8.2 Final test (2024-2026, notebook 07)

All three models are evaluated on the same 3,508 rows. MAE and RMSE are in cases per district-week. Negative percentages mean a smaller error than persistence.

| Model | Test MAE | Test RMSE | MAE vs persistence | RMSE vs persistence |
|---|---|---|---|---|
| Persistence | 12.63 | 29.93 | - | - |
| Random Forest | **12.53** | 33.63 | **-0.8%** | +12.4% |
| XGBoost | 13.22 | 39.19 | +4.6% | +30.9% |

| Test year | Mean cases per district-week | Persistence MAE | Random Forest MAE | XGBoost MAE |
|---|---|---|---|---|
| 2024 | 37.1 | 10.45 | 9.53 | 9.47 |
| 2025 | 37.6 | 8.68 | 8.31 | 8.27 |
| 2026 (to August) | 98.9 | 22.93 | 24.66 | 27.82 |

- By MAE, the Random Forest is the best model in 15 of the 26 reporting areas, persistence in 6 and XGBoost in 5. The Random Forest beats XGBoost in 18 areas.
- The 37 district-weeks with more than 500 cases (1.1% of the test rows) account for 68% of XGBoost's squared error, 63% of the Random Forest's and 50% of persistence's.

**Sensitivity check (duplicated weeks).** On the 52 test rows whose target week is an exact copy of the current week, persistence scores a perfect error of 0. Without the 130 test rows that use a duplicated week, the MAE changes are: Random Forest -4.0% vs persistence, XGBoost +2.1%. Both ensembles are still worse on RMSE. The official results above include all rows.

### 8.3 Experimental hybrid: Random Forest + ARIMA (notebook 08)

An ARIMA(1,0,0) model predicts next week's Random Forest error from the recent errors. It is fitted only on out-of-sample 2023 errors, applied week by week, uses only errors already known at prediction time, and is never refitted on the test weeks.

| Model | Test MAE | Test RMSE |
|---|---|---|
| Random Forest | 12.53 | 33.63 |
| RF + ARIMA | 12.96 | 35.27 |

This is a **negative result**. The Random Forest's errors are almost uncorrelated from one week to the next (median lag-1 autocorrelation about 0 in 2023), so the ARIMA correction mostly adds noise. The hybrid is not used in the dashboard.

## 9. Key findings

1. **Random Forest generalised better than XGBoost** on the unseen years: it had lower MAE and RMSE overall and a much lower RMSE in the 2026 outbreak. XGBoost's tiny validation lead did not carry over, which shows why an untouched test set is needed.
2. **Neither ensemble clearly beats persistence.** Both improve on it in the moderate years 2024-2025 (MAE about 4-9% lower), but both are worse during the 2026 outbreak and have a higher overall RMSE.
3. **Why XGBoost does worse:** only 99 training weeks had more than 500 cases, and 69 of them come from the 2017 epidemic. In 2017, large weekly jumps at high levels were followed by even higher weeks. In July 2026, Colombo went from 1,264 to 1,625 cases, and XGBoost predicted about 2,265 against an actual 1,330. The jump feature (`weekly_case_change`) alone added about +759 cases. Boosting adds up the contributions of correlated features, so a pattern learned from a few rows gets amplified. The Random Forest averages at least 10 past rows per leaf and stays within past values, so it overshoots less.
4. **What the models use:** both depend mostly on `current_week_cases` and the 4-week moving average, with permutation importance measured on the test data. They have essentially learned a smoothed persistence rule, which is why they are so close to it.

## 10. Methodology checks

| Check | Result |
|---|---|
| Data leakage through interpolated weeks | Prevented: rows with an interpolated current or target week are removed (52 training rows, 106 test rows). |
| Target in the features / target alignment | Not in the features (asserted); the target is exactly the following week (asserted in notebooks 03 and 07). |
| Temporal split / train-test contamination | Chronological, with `train < validation < test` asserted. Thresholds and settings use data up to 2023 only, and the test set is used once. |
| Reporting-area coding | Alphabetical codes from notebook 02, used identically by the notebooks and the dashboard. |
| Dashboard vs training features | The dashboard rebuilds the 11 features with the notebook 03 formulas; checked identical on all 3,614 test rows. Dashboard predictions are bit-identical to the evaluation predictions. |
| Reproducibility | Retraining XGBoost gives the saved model's predictions exactly. Random Forest trees are identical across runs; parallel prediction differed by about 1e-13, so the saved model predicts in a single thread (`n_jobs=1`) for exact repeatability. Notebook 07 also reproduces notebook 05's Random Forest validation MAE (18.139). |
| Metrics | MAE and RMSE are pooled over all district-weeks, so they are dominated by the large areas; per-area, per-year and outbreak-week results are also reported. |
| Duplicated weeks | Favour persistence; reported as a sensitivity check without changing the test set. |

## 11. Limitations

- The test period is a single, short period (2024 to August 2026) dominated by one large outbreak, and there is no statistical significance test of the differences.
- Settings were chosen on one validation year from a few candidates, so the validation scores are slightly optimistic.
- Only past case counts are used. There are no weather, mobility or serotype data, which are the kind of information that could signal turning points.
- The most recent 2026 weeks may be incomplete (late reports), and the duplicated weeks may hide data errors.
- The risk thresholds are fixed percentiles from 2007-2023 and do not adjust to population or reporting changes.

## 12. Research note: Random Forest, XGBoost, LightGBM and CatBoost

All four are ensembles of decision trees. Random Forest uses **bagging**; the other three are **gradient boosting** libraries.

| | Random Forest | XGBoost | LightGBM | CatBoost |
|---|---|---|---|---|
| Ensemble type | bagging + random feature subsets (Breiman, 2001) | gradient boosting with a regularised objective and second-order gradients (Chen & Guestrin, 2016) | gradient boosting with histogram-based splits, leaf-wise tree growth, GOSS sampling and exclusive feature bundling (Ke et al., 2017) | gradient boosting with ordered boosting and symmetric (oblivious) trees (Prokhorenkova et al., 2018) |
| Strengths | few settings, robust defaults, hard to overfit badly, trees train in parallel | high accuracy on tabular data, L1/L2 regularisation, built-in missing-value handling, widely used | very fast and memory-efficient on large datasets, native categorical support | strong default settings, built-in handling of categorical features via ordered target statistics, reduces target leakage in boosting, fast prediction |
| Limitations | cannot extrapolate beyond training targets, large model files, impurity importance is biased | more settings to tune, sequential training, can fit rare patterns strongly (seen in this project) | leaf-wise trees can overfit small datasets unless `num_leaves` / `min_data_in_leaf` are limited | training is often slower than LightGBM on large data; its advantages matter most when there are many categorical features |

**Tabular data.** On medium-sized tabular datasets, tree ensembles are still usually as accurate as deep learning or better (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022). Tuned boosting is often the most accurate, while Random Forest is the most robust with little tuning. That matches this project: with about 22,000 rows and noisy outbreak data, the more conservative Random Forest generalised better.

**Image and text data.** Trees split on individual input columns. A single pixel or token carries little meaning on its own, and the important structure is spatial (images) or sequential (text). Deep learning (convolutional networks, transformers) learns that structure directly from raw data and is the standard choice there. Tree ensembles can still be used *on top of* extracted features, such as image embeddings or text TF-IDF vectors and embeddings, but they do not learn the representation themselves.

**Why LightGBM and CatBoost were not run.** Their main advantages are speed on very large datasets and handling of many categorical features. This dataset is small and has a single categorical feature (`district_code`), so they would be expected to behave like XGBoost (boosting), but this was **not tested**. Adding them would also require new dependencies.

**References**
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD 2016*, 785-794.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30*.
- Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems 31*.
- Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. *Neurocomputing*, 50, 159-175.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 2022 Datasets and Benchmarks Track*.
- Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion*, 81, 84-90.

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
  04_baseline_model_training.ipynb            persistence, Decision Tree (validation 2023)
  05_ensemble_model_training.ipynb            Random Forest, XGBoost (validation 2023)
  06_final_model_evaluation.ipynb             final XGBoost vs persistence (original test result)
  07_final_rf_vs_xgboost_comparison.ipynb     final Random Forest vs XGBoost vs persistence + diagnostics
  08_hybrid_rf_arima_experiment.ipynb         experimental RF + ARIMA hybrid
models/
  05_selected_model.json                      validation scores and selected settings
  06_final_xgboost_model.json                 final XGBoost model
  07_final_random_forest_model.joblib         final Random Forest model
  07_final_test_results.json                  test scores, sensitivity check, feature importance
  07_test_predictions.csv                     every test row with actual and predicted values
  08_hybrid_rf_arima_results.json             hybrid experiment results
app/dashboard.py                              Streamlit dashboard
```

## 15. Installation and running the dashboard

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

Run the second command from the project folder.

The dashboard has four tabs:
- **Overview:** national statistics and trends.
- **Historical trends:** weekly cases per reporting area, with its risk thresholds.
- **Prediction & risk:** next-week prediction with a choice of **Random Forest or XGBoost**, risk level, and a table for all 26 areas.
- **Model comparison:** validation and test results, actual vs predicted, feature importance and an interpretation.

The data ends with the week starting 2026-08-29, so the forecast is for the following week, not for today.

**Tested with:** Python 3.14.6, pandas 3.0.3, numpy 2.4.6, scikit-learn 1.9.0, xgboost 3.4.1, statsmodels 0.14.6, streamlit 1.58.0, plotly 6.7.0. The Random Forest file is a joblib/pickle file: load it with the same scikit-learn version, and only load it from a trusted source.

## 16. Reproducing the results

Run the notebooks in order (01 to 08) from inside the `notebooks analysis/` folder, because they use relative paths such as `../dataset`:

```bash
cd "notebooks analysis"
jupyter nbconvert --to notebook --execute --inplace 07_final_rf_vs_xgboost_comparison.ipynb
jupyter nbconvert --to notebook --execute --inplace 08_hybrid_rf_arima_experiment.ipynb
```

- Notebooks 01-03 rewrite the files in `dataset/processed/`, and notebooks 05-08 write to `models/`.
- All models use `random_state = 42`. Notebook 07 checks that it reproduces notebook 05's Random Forest validation score and notebook 06's saved XGBoost model.
- Notebook 08 needs `statsmodels`. The dashboard does not.
