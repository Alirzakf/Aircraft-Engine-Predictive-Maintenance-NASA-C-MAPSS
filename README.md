# Aircraft Engine Predictive Maintenance with NASA C-MAPSS

A reproducible machine learning project for estimating the Remaining Useful Life (RUL) of turbofan engines from multivariate sensor histories in the NASA C-MAPSS FD001 dataset.

The repository covers the full modeling workflow: data validation, target construction, leakage-safe temporal feature engineering, engine-level validation, model comparison, controlled ablation studies, one-time evaluation on the official test set, model persistence, automated testing, and post-hoc interpretation with permutation importance and SHAP.

![Python](https://img.shields.io/badge/Python-3.9-blue)
[![Tests](https://github.com/Alirzakf/Aircraft-Engine-Predictive-Maintenance-NASA-C-MAPSS/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Alirzakf/Aircraft-Engine-Predictive-Maintenance-NASA-C-MAPSS/actions/workflows/tests.yml)
![Tests](https://img.shields.io/badge/tests-28%20passed-success)
![Dataset](https://img.shields.io/badge/dataset-C--MAPSS%20FD001-informational)
![License](https://img.shields.io/badge/license-PolyForm%20Noncommercial-orange)

> [!IMPORTANT]
> This repository is a research and portfolio project. It is not certified for aviation, safety-critical, regulatory, or production use.

## Contents

- [Project summary](#project-summary)
- [Key results](#key-results)
- [Scientific evaluation protocol](#scientific-evaluation-protocol)
- [Dataset](#dataset)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Running the project](#running-the-project)
- [Methodology](#methodology)
- [Target definition](#target-definition)
- [Sensor selection](#sensor-selection)
- [Leakage-safe feature engineering](#leakage-safe-feature-engineering)
- [Validation strategy](#validation-strategy)
- [Model comparison](#model-comparison)
- [Ablation studies](#ablation-studies)
- [Official test evaluation](#official-test-evaluation)
- [Model interpretability](#model-interpretability)
- [Saved model artifact](#saved-model-artifact)
- [Testing and continuous integration](#testing-and-continuous-integration)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Citation](#citation)
- [License](#license)

## Project summary

Predictive maintenance aims to estimate how long an asset can continue operating before failure. In this project, each engine is represented by a time-ordered sequence of operating cycles, operational settings, and sensor readings. The supervised learning target is the number of cycles remaining before the end of the engine trajectory.

The work is built around five practical requirements:

1. **No future-data leakage.** Historical features use only the current and preceding cycles of the same engine.
2. **Engine-level validation.** Complete engines, rather than individual rows, are assigned to either training or validation.
3. **Controlled experiments.** Target definitions and temporal-window sizes are compared while the rest of the pipeline remains fixed.
4. **Test-set discipline.** The final model is selected before the official NASA test set is evaluated.
5. **Reproducibility.** The repository includes reusable source modules, unit tests, continuous integration, a persisted model bundle, and recorded environment metadata.

The final pipeline uses:

- NASA C-MAPSS subset **FD001**
- a capped RUL target with a maximum of **125 cycles**
- **10 selected sensors**
- raw values plus four causal history features per sensor
- the operating cycle as an additional predictor
- **51 total model features**
- a **10-cycle rolling window**
- an **EMA span of 10 cycles**
- a `HistGradientBoostingRegressor` selected on engine-level validation

## Key results

### Final validation result

The selected Histogram Gradient Boosting configuration achieved the lowest validation MAE among the compared models:

| Model | MAE | RMSE | $R^2$ |
|---|---:|---:|---:|
| **Histogram Gradient Boosting, constrained** | **7.854** | 11.744 | 0.921 |
| XGBoost, constrained | 7.913 | **11.571** | **0.923** |
| Random Forest | 8.415 | 11.575 | 0.923 |
| Ridge Regression | 12.817 | 16.047 | 0.852 |
| Dummy Mean | 36.930 | 41.721 | -0.000 |

Histogram Gradient Boosting was selected because it produced the lowest validation MAE, the primary model-selection metric used in the project. The differences among the three ensemble models were small, so the result should not be interpreted as evidence that one algorithm is universally superior.

### Official NASA FD001 test result

After all modeling decisions had been frozen, the selected model was retrained on the complete FD001 training set and evaluated once on the official test trajectories.

| Evaluation target | MAE | RMSE | $R^2$ |
|---|---:|---:|---:|
| **Capped official RUL, maximum 125** | **10.116** | **14.107** | **0.876** |
| Raw official RUL | 11.186 | 15.197 | 0.866 |

The persisted model bundle reproduces these official-test results after being reloaded from disk.

## Scientific evaluation protocol

The official test set was not used for model selection, feature selection, target selection, temporal-window selection, or hyperparameter tuning.

The sequence was:

1. define and validate the preprocessing pipeline;
2. create an engine-level train-validation split;
3. compare baseline and ensemble models on validation data;
4. run target and temporal-window ablations on the same fixed split;
5. select Histogram Gradient Boosting using validation MAE;
6. freeze the modeling configuration;
7. retrain the selected model on all 100 training engines;
8. evaluate once on the official NASA test set;
9. save the fitted model and evaluation metadata;
10. interpret the frozen model on the official test endpoints.

Random Forest and XGBoost are also reported on the official test set as descriptive benchmarks. Their test results did not influence the choice of the persisted model.

## Dataset

This project uses the **NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset**, subset **FD001**.

FD001 represents a single operating condition and a single fault mode. Each row contains:

- an engine identifier;
- the current operating cycle;
- three operational settings;
- 21 sensor measurements.

### Training data

The training set contains complete run-to-failure trajectories:

| Property | Value |
|---|---:|
| Rows | 20,631 |
| Engines | 100 |
| Columns | 26 |
| Minimum cycle | 1 |
| Maximum cycle | 362 |
| Missing values | 0 |
| Duplicate engine-cycle rows | 0 |

### Test data

The test file contains partial trajectories for 100 engines. Prediction is performed at the final observed cycle of each test engine. The accompanying `RUL_FD001.txt` file provides one ground-truth RUL value per test engine.

| Property | Value |
|---|---:|
| Test rows | 13,096 |
| Test engines | 100 |
| Official RUL labels | 100 |
| Endpoint feature matrix | $100 \times 51$ |

The raw FD001 files used by the notebooks are located in `data/raw/`.

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── tests.yml
├── data/
│   └── raw/
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
├── models/
│   └── hist_gradient_boosting_fd001.joblib
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_rul_and_feature_engineering.ipynb
│   ├── 03_sensor_analysis.ipynb
│   ├── 04_baseline_modeling.ipynb
│   ├── 05_gradient_boosting_and_optimization.ipynb
│   ├── 06_target_and_window_ablation.ipynb
│   ├── 07_official_test_evaluation.ipynb
│   └── 08_model_interpretability.ipynb
├── outputs/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── rul_builder.py
│   └── sensor_utils.py
├── tests/
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_rul_builder.py
│   └── test_sensor_utils.py
├── CITATION.cff
├── COMMERCIAL_LICENSE.md
├── LICENSE
├── NOTICE
├── README.md
└── requirements.txt
```

### Notebook sequence

| Notebook | Purpose |
|---|---|
| `01_data_loading.ipynb` | Load and validate the FD001 training data. |
| `02_rul_and_feature_engineering.ipynb` | Construct capped RUL and demonstrate causal temporal features. |
| `03_sensor_analysis.ipynb` | Inspect variance, correlation, and degradation patterns; define the candidate sensor set. |
| `04_baseline_modeling.ipynb` | Build the engine-level split and compare Dummy, Ridge, and Random Forest baselines. |
| `05_gradient_boosting_and_optimization.ipynb` | Compare Histogram Gradient Boosting and XGBoost; examine prediction clipping. |
| `06_target_and_window_ablation.ipynb` | Compare linear and capped targets and rolling/EMA windows of 3, 5, 10, and 20 cycles. |
| `07_official_test_evaluation.ipynb` | Retrain frozen models, perform the one-time official evaluation, and persist the selected model bundle. |
| `08_model_interpretability.ipynb` | Reproduce the saved result and analyze the final model with permutation importance and SHAP. |

## Installation

### Requirements

- Python 3.9
- `pip`
- JupyterLab or another Jupyter-compatible environment

The saved artifact records the environment in which it was generated:

| Package | Recorded version |
|---|---:|
| Python | 3.9.6 |
| NumPy | 2.0.2 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |
| joblib | 1.5.3 |
| XGBoost | 2.1.4 |
| SHAP | 0.49.1 |

### Create an environment

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3.9 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the project

Start JupyterLab from the repository root:

```bash
jupyter lab
```

Run the notebooks in numerical order. Each notebook imports shared functionality from `src/` and is designed to be independently understandable while remaining consistent with the full workflow.

Run the automated tests with:

```bash
pytest -v
```

The current suite contains **28 passing tests**.

### Minimal preprocessing example

```python
from pathlib import Path

from src.config import EMA_SPAN, ROLLING_WINDOW, RUL_CAP
from src.data_loader import load_cmapss_data
from src.preprocessing import add_sensor_history_features
from src.rul_builder import add_train_rul

selected_sensors = [
    "s2", "s3", "s4", "s7", "s11",
    "s12", "s15", "s17", "s20", "s21",
]

train_path = Path("data/raw/train_FD001.txt")

train_df = load_cmapss_data(train_path)
train_df = add_train_rul(train_df, rul_cap=RUL_CAP)
train_df = add_sensor_history_features(
    train_df,
    sensors=selected_sensors,
    rolling_window=ROLLING_WINDOW,
    ema_span=EMA_SPAN,
)
```

### Load the persisted model

```python
from pathlib import Path

import joblib

bundle_path = Path("models/hist_gradient_boosting_fd001.joblib")
bundle = joblib.load(bundle_path)

model = bundle["model"]
feature_columns = bundle["feature_columns"]
selected_sensors = bundle["selected_sensors"]

print(bundle["model_name"])
print(bundle["official_test_metrics"])
```

New trajectories must be processed with the same feature definitions and column order stored in the bundle before calling `model.predict(...)`.

## Methodology

The pipeline consists of the following stages:

1. load the raw C-MAPSS text files and assign descriptive column names;
2. validate row structure, engine-cycle uniqueness, cycle continuity, and missing values;
3. construct the training RUL target without retaining final lifetime as a feature;
4. remove near-constant sensors from consideration during exploratory analysis;
5. select a compact set of degradation-related sensors;
6. create causal history features independently for each engine;
7. split the training set by engine ID;
8. compare linear and nonlinear regression models;
9. evaluate target and history-window assumptions through ablation studies;
10. select the final model using validation MAE;
11. retrain on all training engines and evaluate once on official test endpoints;
12. save and reload the fitted model bundle;
13. analyze global and local model behavior.

## Target definition

### Linear RUL

For engine $i$ at cycle $t$, the ordinary linear target is:

$$
\mathrm{RUL}_{i,t}^{\text{linear}} = T_i - t
$$

where $T_i$ is the final observed training cycle of engine $i$.

The final cycle is used only to construct the target. It is not kept as an input feature.

### Capped RUL

The final project target is:

```math
\mathrm{RUL}_{i,t}^{\text{capped}}
=
\min\left(\mathrm{RUL}_{i,t}^{\text{linear}}, 125\right)
```

The cap treats the early healthy region as a plateau rather than asking the model to distinguish among very large RUL values that are weakly supported by degradation signals.

In FD001, **8,031 rows**, or **38.93%** of the training observations, are affected by the cap.

The value 125 is an empirically selected configuration for this project. It is not presented as a universal physical constant.

## Sensor selection

The final sensor set is:

```text
s2, s3, s4, s7, s11, s12, s15, s17, s20, s21
```

Exploratory analysis identified the following sensors as constant or effectively constant in FD001:

```text
s1, s5, s10, s16, s18, s19
```

Sensor `s6` has small but non-zero variance and weak correlation with RUL. It was reviewed separately rather than being removed automatically by the near-constant threshold.

The final selection reflects FD001-specific variance, correlation, observed degradation behavior, and predictive usefulness. It should not be assumed to transfer unchanged to FD002, FD003, or FD004.

## Leakage-safe feature engineering

The original value of every selected sensor is retained. Four additional historical features are created for each selected sensor:

1. one-cycle difference;
2. rolling mean;
3. exponential moving average;
4. expanding mean.

With 10 sensors, this produces 50 sensor-derived features. The operating cycle is included as the 51st model feature.

### One-cycle difference

$$
\Delta x_t = x_t - x_{t-1}
$$

The first difference of each engine is filled with zero because no earlier observation exists for that engine.

### Rolling mean

For a window of width $w$:

```math
\bar{x}_{t}^{(w)}
=
\frac{1}{m_t}
\sum_{k=\max(1,\,t-w+1)}^{t} x_k
```

where $m_t$ is the number of available observations in the partial or full window. The implementation uses `min_periods=1`, so early cycles do not produce missing values.

The selected value is:

$$
w = 10
$$

### Exponential moving average

```math
\mathrm{EMA}_t
=
\alpha x_t + (1-\alpha)\mathrm{EMA}_{t-1}
```

with:

$$
\alpha = \frac{2}{s+1}
$$

The selected EMA span is:

$$
s = 10
$$

### Expanding mean

```math
\bar{x}_{t}^{\text{exp}}
=
\frac{1}{t}
\sum_{k=1}^{t} x_k
```

### Causality guarantees

All history features are:

- computed after sorting by engine ID and cycle;
- calculated separately within each engine;
- based only on the current and earlier cycles;
- prevented from crossing engine boundaries;
- free of final-lifetime columns such as `max_cycle` or `cycle_progress`.

The project assumes **same-cycle prediction**: the sensor values observed at cycle $t$ may be used to estimate RUL at cycle $t$.

## Validation strategy

A fixed grouped holdout is created with `GroupShuffleSplit`:

```text
n_splits = 1
test_size = 0.20
random_state = 42
```

This produces:

- 80 training engines;
- 20 validation engines;
- zero shared engines between the two sets.

The split is performed by engine ID rather than by row. This is important because randomly splitting cycles would place measurements from the same degradation trajectory in both training and validation, leading to an overly optimistic estimate of generalization.

The fixed split is reused across model and ablation experiments so that changes in metrics can be attributed to the factor under study.

## Model comparison

### Dummy mean regressor

The dummy model predicts the mean training target for every row. It defines the minimum useful baseline.

### Ridge regression

Ridge regression is fitted in a pipeline with standardization and `alpha=1.0`. It tests how much of the relationship can be captured by a regularized linear model.

### Random Forest

The Random Forest configuration is:

```text
n_estimators = 200
max_depth = None
min_samples_leaf = 2
max_features = "sqrt"
random_state = 42
```

### Histogram Gradient Boosting

The selected model uses:

```text
learning_rate = 0.05
max_iter = 300
max_leaf_nodes = 31
min_samples_leaf = 20
l2_regularization = 1.0
early_stopping = False
random_state = 42
```

### XGBoost

The XGBoost comparison uses:

```text
objective = "reg:squarederror"
n_estimators = 300
learning_rate = 0.05
max_depth = 6
min_child_weight = 5
subsample = 0.8
colsample_bytree = 0.8
reg_alpha = 0.0
reg_lambda = 1.0
tree_method = "hist"
random_state = 42
```

### Prediction constraints

Because the project target is bounded, constrained predictions are defined as:

```math

\widehat{\mathrm{RUL}}_{\mathrm{constrained}}

=

\min\left(125,\max\left(0,\widehat{\mathrm{RUL}}\right)\right)

```

Clipping produced only small validation improvements. On the official test endpoints, the selected Histogram Gradient Boosting model produced no values below zero or above 125.

## Ablation studies

### Temporal-window ablation with linear RUL

Rolling windows and EMA spans of 3, 5, 10, and 20 cycles were compared with the Random Forest model while keeping the engine split, sensors, and other modeling choices fixed.

| Window and EMA span | MAE | RMSE | $R^2$ |
|---:|---:|---:|---:|
| 3 | 20.988 | 27.808 | 0.821 |
| 5 | 20.502 | 27.197 | 0.828 |
| **10** | **20.146** | **26.702** | **0.835** |
| 20 | 20.286 | 26.914 | 0.832 |

The 10-cycle configuration achieved the lowest MAE and RMSE.

### Target ablation

The linear and capped targets were compared on the same engine split with the same Random Forest architecture and 10-cycle history configuration.

| Experiment | MAE | RMSE | $R^2$ |
|---|---:|---:|---:|
| Linear target, full validation | 20.146 | 26.702 | 0.835 |
| Capped target, capped validation | **8.415** | **11.575** | **0.923** |

A direct full-range comparison favors the capped target partly because it changes the target distribution. For a fairer degradation-region comparison, both trained models were also evaluated where the linear and capped ground truths are identical.

| Training target in the common critical region | MAE | RMSE | $R^2$ |
|---|---:|---:|---:|
| Linear RUL | 14.562 | 20.316 | 0.688 |
| **Capped RUL** | **9.886** | **13.019** | **0.872** |

In this common region, capped-target training improved MAE by **32.11%** and RMSE by **35.92%**.

### Temporal-window consistency check with capped RUL

The window experiment was repeated after adopting the capped target:

| Window and EMA span | MAE | RMSE | $R^2$ |
|---:|---:|---:|---:|
| 3 | 9.132 | 12.607 | 0.909 |
| 5 | 8.755 | 12.124 | 0.916 |
| **10** | **8.415** | **11.575** | **0.923** |
| 20 | 8.721 | 11.851 | 0.919 |

The 10-cycle configuration remained the strongest choice after the target changed.

## Official test evaluation

For official evaluation, the selected feature pipeline is applied to every partial test trajectory. Only the final observed cycle of each test engine is retained, yielding one prediction per engine.

The three ensemble models were retrained on all 20,631 training rows from all 100 training engines. The following table reports both the raw NASA labels and labels capped at the project target maximum.

| Model | Target definition | MAE | RMSE | $R^2$ |
|---|---|---:|---:|---:|
| Random Forest | Raw official RUL | 11.936 | 16.575 | 0.841 |
| Random Forest | Capped official RUL, 125 | 10.866 | 15.418 | 0.852 |
| **Histogram Gradient Boosting** | **Raw official RUL** | **11.186** | **15.197** | **0.866** |
| **Histogram Gradient Boosting** | **Capped official RUL, 125** | **10.116** | **14.107** | **0.876** |
| XGBoost | Raw official RUL | 11.393 | 15.639 | 0.858 |
| XGBoost | Capped official RUL, 125 | 10.323 | 14.575 | 0.868 |

The selected model also produced the lowest error among the retained benchmark models on this particular test set. That agreement supports the earlier validation decision, but it was not used to make or revise the selection.

## Model interpretability

Notebook 08 reloads the persisted model, reconstructs the official test feature matrix, verifies the stored MAE, and applies two complementary interpretation methods.

### Permutation importance

Permutation importance measures the increase in MAE when a feature is randomly disrupted. The leading features include:

| Rank | Feature | Mean importance |
|---:|---|---:|
| 1 | `cycle` | 2.973 |
| 2 | `s4_ema_10` | 2.906 |
| 3 | `s12_expanding_mean` | 2.237 |
| 4 | `s20_expanding_mean` | 1.904 |
| 5 | `s15_rolling_mean_10` | 1.732 |
| 6 | `s11_ema_10` | 1.423 |
| 7 | `s4_rolling_mean_10` | 1.210 |
| 8 | `s11_rolling_mean_10` | 1.187 |
| 9 | `s2_rolling_mean_10` | 1.093 |
| 10 | `s21_rolling_mean_10` | 1.022 |

### SHAP analysis

SHAP values are computed for all 100 official test endpoints and all 51 model features.

The highest mean absolute SHAP values are:

| Rank | Feature | Mean absolute SHAP value |
|---:|---|---:|
| 1 | `cycle` | 5.776 |
| 2 | `s4_ema_10` | 5.542 |
| 3 | `s12_expanding_mean` | 4.360 |
| 4 | `s15_rolling_mean_10` | 4.295 |
| 5 | `s20_expanding_mean` | 4.174 |
| 6 | `s4_rolling_mean_10` | 3.708 |
| 7 | `s11_ema_10` | 3.184 |
| 8 | `s11_rolling_mean_10` | 3.095 |
| 9 | `s21_rolling_mean_10` | 2.845 |
| 10 | `s2_rolling_mean_10` | 2.675 |

The two methods produce broadly consistent rankings. Most leading predictors are rolling, exponential, or expanding summaries rather than isolated raw sensor values, indicating that the model relies strongly on accumulated degradation trends.

### Local explanations

The notebook includes local SHAP explanations for:

- the test engine with the smallest absolute error;
- the engine closest to the median absolute error;
- the engine with the largest absolute error.

The corresponding examples are:

| Example | Engine | Actual RUL | Predicted RUL | Absolute error |
|---|---:|---:|---:|---:|
| Smallest error | 77 | 34 | 33.977 | 0.023 |
| Median-like error | 85 | 118 | 124.092 | 6.092 |
| Largest error | 45 | 114 | 65.234 | 48.766 |

### SHAP numerical check

Strict additivity checking was disabled for the scikit-learn Histogram Gradient Boosting model because SHAP produced a numerical mismatch. The notebook therefore reconstructs predictions explicitly from the SHAP base values and feature attributions.

Recorded absolute additivity errors were:

| Statistic | Error |
|---|---:|
| Maximum | 1.409 |
| Mean | 0.052 |
| Median | 0.007 |

The discrepancy is documented rather than hidden and should be kept in mind when interpreting individual explanations.

## Saved model artifact

The final model is stored at:

```text
models/hist_gradient_boosting_fd001.joblib
```

The joblib bundle contains:

- the fitted `HistGradientBoostingRegressor`;
- the model name and full estimator parameters;
- the ordered list of 51 feature columns;
- the 10 selected sensors;
- the target definition and cap;
- rolling-window and EMA configuration;
- historical feature names;
- training row, feature, and engine counts;
- official capped and raw test metrics;
- model-selection metadata confirming that the official test was not used for selection;
- Python and package versions;
- a UTC creation timestamp.

The saved model is reloaded and checked against the in-memory predictions. The reloaded artifact reproduces:

```text
Capped official RUL
MAE  = 10.116
RMSE = 14.107
R²   = 0.876

Raw official RUL
MAE  = 11.186
RMSE = 15.197
R²   = 0.866
```

## Testing and continuous integration

The unit tests cover the reusable data and preprocessing modules. They verify behavior including:

- standard C-MAPSS column naming;
- train, test, and RUL file loading;
- missing-file errors;
- required-column validation;
- linear and capped RUL construction;
- final-cycle RUL equal to zero;
- invalid cap handling;
- row-count preservation;
- deterministic feature generation;
- engine-boundary isolation;
- no future-cycle dependence in historical features;
- preservation of raw sensor values;
- rolling, EMA, difference, and expanding feature behavior;
- sensor variance calculations;
- near-constant sensor detection;
- invalid threshold handling.

The current test run is:

```text
28 passed
```

GitHub Actions runs the suite on:

- pushes to `main`;
- pull requests targeting `main`.

The workflow checks out the repository, installs Python 3.9 and the project dependencies, and runs:

```bash
pytest -v
```

## Reproducibility

Reproducibility is supported at several levels:

- constants are centralized in `src/config.py`;
- data loading, RUL construction, preprocessing, and variance analysis are implemented in reusable modules;
- the validation split uses a fixed random seed;
- model random states are fixed;
- experiments reuse the same engine partition;
- all candidate models use the same feature matrix and target definition within each comparison;
- the final feature order is saved with the model;
- the persisted artifact records package versions;
- the saved model is reloaded and its official metrics are recomputed;
- automated tests run locally and in GitHub Actions.

The dependency file uses minimum compatible versions for most packages and an exact XGBoost version because the project was executed with XGBoost 2.1.4. The recorded model environment should be preferred when reproducing the saved artifact exactly.

## Limitations

### Single subset

The project currently covers only FD001, which contains one operating condition and one fault mode. Results should not be generalized to the more heterogeneous C-MAPSS subsets without new analysis.

### Single grouped holdout

Model development uses one fixed 80/20 engine-level split. This supports clean and repeatable comparison, but grouped cross-validation would provide a more complete estimate of variation across engine partitions.

### Simulated data

C-MAPSS is a simulation benchmark. Performance on FD001 does not establish readiness for real aircraft engines or other industrial assets.

### Capped target

The 125-cycle cap improves this project's validation behavior and critical-region performance, but it changes the learning target and reflects a modeling assumption. Applications that require accurate estimates far from failure may prefer another target formulation.

### Point predictions only

The model returns a single RUL estimate and does not quantify predictive uncertainty or produce calibrated intervals.

### Test-set size

The official FD001 evaluation contains 100 engines. Model rankings that differ by small margins should therefore be interpreted cautiously.

### Interpretation is associative

Permutation importance and SHAP describe how the trained model uses its features. They do not prove physical causality or identify the true mechanical cause of degradation.

### SHAP approximation mismatch

The local SHAP explanations have small reconstruction discrepancies for the saved Histogram Gradient Boosting estimator. These are measured in Notebook 08 and should not be ignored when reading individual attributions.

## Future work

Reasonable extensions include:

- repeat the workflow on FD002, FD003, and FD004;
- add grouped cross-validation and report metric distributions;
- evaluate the asymmetric NASA scoring function alongside MAE, RMSE, and $R^2$;
- compare direct endpoint training with the current cycle-level training strategy;
- add uncertainty estimates or conformal prediction intervals;
- study condition normalization for multi-condition subsets;
- compare tree ensembles with sequence models under the same leakage controls;
- test reduced feature sets suggested by permutation importance and SHAP;
- export interpretability figures and experiment tables as versioned report artifacts;
- add a small inference interface that validates schema, feature order, and preprocessing metadata before prediction.

## AI Assistance Disclosure

AI-assisted tools were used as part of the development workflow for code review, debugging support, documentation editing, and discussion of modeling and experimental-design choices.

The project architecture, data-processing pipeline, leakage controls, feature engineering, model training, validation strategy, official test evaluation, interpretability analysis, and reported results were implemented, executed, reviewed, and verified by the author. Suggestions produced by AI tools were treated as review input rather than as authoritative output and were tested or revised before inclusion in the repository.

## Citation

Citation metadata is provided in `CITATION.cff`.

```bibtex
@software{kafi_2026_cmapss,
  author  = {Alireza Kafi},
  title   = {Aircraft Engine Predictive Maintenance using NASA C-MAPSS},
  year    = {2026},
  url     = {https://github.com/Alirzakf/Aircraft-Engine-Predictive-Maintenance-NASA-C-MAPSS}
}
```

Users of the dataset should also cite the original NASA C-MAPSS source according to the dataset provider's citation guidance.

## License

Copyright © 2026 Alireza Kafi.

The repository is released under the **PolyForm Noncommercial License 1.0.0** for permitted noncommercial use, including personal study, research, experimentation, and education.

Commercial, industrial, internal business, product, service, consulting, or production use requires a separate written commercial license. See:

- `LICENSE`
- `NOTICE`
- `COMMERCIAL_LICENSE.md`

A commercial license does not certify the software for aviation, safety-critical, regulatory, or mission-critical operation.
