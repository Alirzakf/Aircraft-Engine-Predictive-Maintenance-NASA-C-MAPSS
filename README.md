# Aircraft Engine Predictive Maintenance using NASA C-MAPSS

A modular machine learning project for predicting the Remaining Useful
Life (RUL) of aircraft turbofan engines using the NASA C-MAPSS
degradation dataset.

The project focuses on reproducible experimentation, leakage-safe
feature engineering, engine-level validation, interpretable model
comparison, and scientifically justified modeling decisions.

![Tests](https://github.com/<username>/<repo>/actions/workflows/tests.yml/badge.svg)

![Python](https://img.shields.io/badge/Python-3.9-blue)

![Status](https://img.shields.io/badge/status-active-success)

> **License Notice**
>
> This repository is available for **noncommercial research, education,
> personal study, experimentation, and academic citation**.
>
> Commercial, industrial, or production use requires a separate written
> commercial license from **Alireza Kafi**.

---

# Project Objectives

The primary objectives of this project are:

- Predict Remaining Useful Life (RUL) of aircraft turbofan engines.
- Build a modular and reusable machine learning pipeline.
- Prevent future-data leakage throughout preprocessing.
- Compare classical regression and ensemble learning models.
- Evaluate important modeling assumptions using controlled ablation
  studies.
- Produce a reproducible research-oriented implementation suitable for
  academic work and portfolio presentation.

---

# Dataset

This project uses the **NASA C-MAPSS Turbofan Engine Degradation
Simulation Dataset**.

Current subset:

- **FD001**

The training data contain complete run-to-failure trajectories for
simulated aircraft turbofan engines.

The official NASA test trajectories and ground-truth Remaining Useful
Life labels will be used during the final evaluation stage of the
project.

---

# Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── tests.yml
├── data/
│   └── raw/
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_rul_and_feature_engineering.ipynb
│   ├── 03_sensor_analysis.ipynb
│   ├── 04_baseline_modeling.ipynb
│   ├── 05_gradient_boosting_and_optimization.ipynb
│   └── 06_target_and_window_ablation.ipynb
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
├── LICENSE
├── COMMERCIAL_LICENSE.md
├── README.md
└── requirements.txt
```

---

# Current Project Status

## Completed

- NASA C-MAPSS FD001 data loading
- Data validation
- Leakage-safe RUL construction
- Linear and capped RUL comparison
- Historical feature engineering
- Sensor variance analysis
- Sensor selection
- Engine-level train-validation split
- Dummy baseline
- Ridge Regression
- Random Forest
- Histogram Gradient Boosting
- XGBoost
- Rolling-window ablation
- RUL target ablation
- Prediction-range constraint analysis
- Automated unit testing
- Continuous Integration using GitHub Actions

## Next Stage

The next milestone is evaluation on the official NASA FD001 test set.

---

# Final Modeling Configuration

The final modeling pipeline was selected through controlled experiments
rather than arbitrary parameter choices.

## Remaining Useful Life Target

The original linear Remaining Useful Life target is

$$
RUL_{linear}=T_i-t
$$

where:

- $T_i$ is the final observed cycle of engine $i$
- $t$ is the current operating cycle

The final project uses a capped target:

$$
RUL_{capped}
=
\min(RUL_{linear},125)
$$

The cap represents an early healthy operating region where sensor
measurements contain little information for distinguishing very large
Remaining Useful Life values.

The selected cap of **125 cycles** should be interpreted as an
empirically supported project configuration rather than a universal
physical constant.

## Historical Context

The following temporal windows were evaluated:

- 3 cycles
- 5 cycles
- 10 cycles
- 20 cycles

A 10-cycle rolling window together with a 10-cycle EMA span produced
the strongest validation performance.

The experiment was repeated after adopting the finalized capped RUL
target.

The 10-cycle configuration remained the best-performing option under
both target definitions.

## Selected Sensors

The final modeling pipeline uses:

```text
s2
s3
s4
s7
s11
s12
s15
s17
s20
s21
```

The selected sensors were chosen after exploratory analysis of sensor
variance, degradation behaviour, and predictive usefulness.

The selection is specific to the FD001 operating condition and should
not automatically be generalized to the remaining NASA C-MAPSS
subsets.

---

# Feature Engineering

The project preserves the original sensor measurements and creates
multiple historical features for every selected sensor.

The generated features are:

- raw sensor value
- first-order difference
- rolling mean
- exponential moving average (EMA)
- expanding mean

## First Difference

The first-order difference measures the short-term change between
consecutive operating cycles.

$$
\Delta x_t=x_t-x_{t-1}
$$

This feature captures sudden sensor changes that may indicate the
beginning of degradation.

---

## Rolling Mean

The rolling mean estimates the recent local operating trend.

$$
\bar{x}_t^{(w)}
=
\frac1{m_t}
\sum_{k=\max(1,t-w+1)}^{t}x_k
$$

where

- $begin:math:text$w$end:math:text$ is the rolling window size,
- $begin:math:text$m\_t$end:math:text$ is the number of available observations inside the window.

The finalized project uses

$$
w=10
$$

This value was selected through the controlled temporal-window
ablation study presented in Notebook 06.

---

## Exponential Moving Average

The exponential moving average assigns larger weights to more recent
observations.

$$
EMA_t
=
\alpha x_t
+
(1-\alpha)EMA_{t-1}
$$

where

$$
\alpha=\frac{2}{s+1}
$$

The finalized project configuration uses

$$
s=10
$$

Compared with the rolling mean, EMA reacts more quickly to recent
changes while still reducing sensor noise.

---

## Expanding Mean

The expanding mean summarizes the cumulative operating history.

$$
\bar{x}_t^{exp}
=
\frac1t
\sum_{k=1}^{t}x_k
$$

Unlike the rolling mean, every previous observation contributes to the
estimate.

This feature provides a gradually stabilizing baseline describing the
historical behaviour of each engine.

---

## Leakage-Safe Feature Engineering

All historical features satisfy the following constraints:

- computed independently for each engine,
- use only current and previous observations,
- never use future operating cycles,
- never cross engine boundaries.

The implementation is therefore suitable for online Remaining Useful
Life prediction where future measurements are unavailable.

---

# Data Leakage Protection

Preventing future-data leakage is one of the primary goals of this
project.

Several safeguards are implemented throughout the preprocessing
pipeline.

## Target Leakage Prevention

The final observed engine lifetime is used **only** to construct the
training target.

The lifetime information is never retained as an input feature.

---

## Historical Features

Rolling statistics, exponential moving averages and expanding means are
computed independently for every engine.

Only historical observations are used.

---

## Engine Separation

Training and validation datasets are separated using engine IDs.

Entire engines belong either to the training set or to the validation
set.

No operating cycle from the same engine appears in both subsets.

This prevents the model from memorizing degradation trajectories from
engines already observed during training.

---

## Automated Verification

Unit tests verify that

- historical features never depend on future observations,
- feature engineering preserves row counts,
- feature engineering does not modify raw sensor values,
- engine boundaries are respected,
- capped RUL behaves correctly,
- preprocessing functions remain deterministic.

---

# Validation Strategy

All experiments use a fixed engine-level validation split.

The validation protocol keeps the following components unchanged:

- selected sensors,
- feature engineering,
- Random State,
- preprocessing,
- evaluation metrics.

Only the modeling component under investigation changes.

This design ensures that differences in predictive performance are
caused by the model itself rather than by unrelated experimental
changes.

The current validation split uses

```text
random_state = 42
```

Grouped cross-validation is intentionally reserved for future work.

---

# Models

The following regression models have been evaluated.

## Dummy Mean Regressor

Provides a constant baseline prediction.

This establishes the minimum level of performance expected from any
useful predictive model.

---

## Ridge Regression

Introduces a linear baseline with L2 regularization.

This model evaluates whether a linear relationship between engineered
features and Remaining Useful Life is sufficient.

---

## Random Forest

Random Forest provides the strongest baseline ensemble model.

Its nonlinear decision trees naturally capture complex degradation
patterns without requiring feature scaling.

---

## Histogram Gradient Boosting

Histogram Gradient Boosting is evaluated as an efficient gradient
boosting implementation available within scikit-learn.

The experiment uses the finalized feature engineering pipeline and the
same validation split adopted by every previous model.

---

## XGBoost

XGBoost provides an additional state-of-the-art boosting algorithm.

The implementation uses the same dataset, target definition, features,
validation split and evaluation metrics as every previous experiment,
allowing a fair comparison between ensemble learning approaches.

---

# Validation Results

The following results use:

- capped Remaining Useful Life target,
- 10-cycle rolling window,
- 10-cycle EMA span,
- identical selected sensors,
- identical engine-level validation split.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| HistGradientBoosting (constrained) | **7.854** | 11.744 | 0.921 |
| XGBoost (constrained) | 7.913 | **11.571** | **0.923** |
| Random Forest | 8.415 | 11.575 | 0.923 |
| Ridge Regression | 12.817 | 16.047 | 0.852 |
| Dummy Mean | 36.930 | 41.721 | -0.000 |

Histogram Gradient Boosting achieved the lowest Mean Absolute Error.

XGBoost achieved the numerically lowest RMSE and highest R², although
its performance difference from Random Forest is negligible.

The narrow performance differences among the leading ensemble models
suggest that the finalized feature engineering pipeline contributes
more to predictive performance than the choice between modern ensemble
algorithms.

---

# Prediction-Range Constraint

The finalized Remaining Useful Life target is bounded between

$$
0 \le RUL \le 125
$$

Both Histogram Gradient Boosting and XGBoost occasionally produced
predictions slightly above the upper target boundary.

Instead of hiding this behaviour, both raw and constrained predictions
are reported.

The constrained prediction is

$$
\hat{RUL}_{constrained}
=
\min
\left(
125,
\max(0,\hat{RUL})
\right)
$$

The constraint produced only small improvements in predictive
performance, indicating that the models already respected the target
range in almost all cases.

---

# Window Ablation Results

One of the primary design questions in this project is determining the
appropriate amount of historical information used for feature
engineering.

Rather than selecting the rolling window arbitrarily, four candidate
values were evaluated while keeping every other component fixed.

The following values were tested:

- 3 cycles
- 5 cycles
- 10 cycles
- 20 cycles

The Random Forest architecture, selected sensors, engine-level
validation split, and evaluation metrics remained unchanged.

## Linear RUL Target

| Window / EMA Span | MAE | RMSE | R² |
|---:|---:|---:|---:|
| 3 | 20.988 | 27.808 | 0.821 |
| 5 | 20.502 | 27.197 | 0.828 |
| **10** | **20.146** | **26.702** | **0.835** |
| 20 | 20.286 | 26.914 | 0.832 |

The 10-cycle configuration achieved the best validation performance.

---

## Final Capped-Target Consistency Check

After adopting the finalized capped Remaining Useful Life target, the
window-selection experiment was repeated to verify that the chosen
configuration remained optimal.

| Window / EMA Span | MAE | RMSE | R² |
|---:|---:|---:|---:|
| 3 | 9.132 | 12.607 | 0.909 |
| 5 | 8.755 | 12.124 | 0.916 |
| **10** | **8.415** | **11.575** | **0.923** |
| 20 | 8.721 | 11.851 | 0.919 |

The ranking remained unchanged.

This consistency check confirms that the selected temporal
configuration is not dependent only on the original linear-target
experiment.

---

# Target Ablation Results

The project compares two Remaining Useful Life target definitions.

## Linear Target

The conventional target is

$$
RUL_{linear}=T_i-t
$$

where Remaining Useful Life decreases linearly throughout the entire
engine lifetime.

---

## Capped Target

The finalized project uses

$$
RUL_{capped}
=
\min(RUL_{linear},125)
$$

Approximately **39%** of the FD001 training observations are affected
by the cap.

The cap reduces unnecessary learning during the early healthy operating
region where sensor measurements provide little information for
distinguishing large Remaining Useful Life values.

---

## Critical-Region Comparison

Both target definitions were evaluated on the same critical degradation
region where they represent identical ground truth.

| Training Target | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear RUL | 14.562 | 20.316 | 0.688 |
| **Capped RUL** | **9.886** | **13.019** | **0.872** |

Training with the capped target improved:

- MAE by approximately **32.11%**
- RMSE by approximately **35.92%**

These improvements justify adopting the capped Remaining Useful Life
target throughout the remainder of the project.

---

# Automated Testing

The repository includes automated unit tests covering:

- dataset loading,
- dataset column definitions,
- missing-file handling,
- Remaining Useful Life construction,
- capped-target behaviour,
- input immutability,
- historical feature engineering,
- leakage prevention,
- engine-boundary preservation,
- configurable feature names,
- sensor variance utilities,
- threshold validation.

Run the complete test suite using

```bash
pytest -v
```

Current project status:

```text
28 passed
```

---

# Continuous Integration

GitHub Actions automatically performs the following tasks whenever code
is pushed to the repository:

1. Check out the repository.
2. Install Python.
3. Install project dependencies.
4. Execute the complete pytest suite.

This guarantees that future modifications do not silently break the
project.

---

# Installation

## Clone the repository

```bash
git clone <repository-url>
cd nasa-turbofan-predictive-maintenance
```

---

## Create a virtual environment

Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

---

## Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Run automated tests

```bash
pytest -v
```

---

## Launch JupyterLab

```bash
jupyter lab
```

---

## macOS Requirement

XGBoost requires the OpenMP runtime.

Install it using Homebrew.

```bash
brew install libomp
```

---

# Reproducibility

The project emphasizes reproducible experimentation.

The following elements are fixed throughout model comparison:

- deterministic random seed,
- centralized configuration,
- reusable preprocessing modules,
- engine-level validation split,
- automated testing,
- continuous integration.

The primary configuration values are

```python
ROLLING_WINDOW = 10
EMA_SPAN = 10
RUL_CAP = 125
NEAR_CONSTANT_VARIANCE_THRESHOLD = 1e-10
```

These values are stored centrally inside

```text
src/config.py
```

to ensure consistent behaviour across all notebooks.

---

# Current Limitations

The current implementation intentionally focuses on the offline
model-development stage.

Current limitations include:

- official NASA FD001 test evaluation has not yet been completed,
- validation currently uses one engine-level split,
- rolling-window and EMA-span values were optimized together,
- extensive hyperparameter optimization has been intentionally deferred,
- feature importance should not be interpreted as causal inference,
- current-cycle sensor measurements are assumed to be available before
  prediction,
- the implementation targets offline batch inference rather than
  streaming deployment.

---

# Future Work

Planned future developments include:

- official NASA FD001 test evaluation,
- grouped cross-validation,
- Bayesian or Optuna hyperparameter optimization,
- SHAP explainability,
- uncertainty estimation,
- experiment tracking,
- model serialization,
- streaming inference,
- deployment pipeline,
- Docker containerization.

---

# Citation

If this repository contributes to your research, academic work,
teaching material, publication, or educational content, please cite it.

A machine-readable `CITATION.cff` file will be included in future
releases.

Suggested citation:

```text
Kafi, Alireza.
Aircraft Engine Predictive Maintenance using NASA C-MAPSS.
GitHub Repository.
2026.
```

Citation is appreciated because it helps support continued development
and future research on this project.

---

# License

Copyright (c) 2026 Alireza Kafi

This repository is distributed under the **PolyForm Noncommercial
License 1.0.0**.

The license permits noncommercial use including:

- academic research,
- education,
- personal learning,
- experimentation,
- benchmarking,
- citation,
- noncommercial derivative works.

Commercial or industrial use is **not** permitted under the public
license.

A separate commercial license must be obtained before using this
project in any commercial environment.

For the complete legal terms, see the `LICENSE` file.

---

# Commercial Licensing

Organizations wishing to use this project commercially are encouraged
to contact the author.

Examples include (but are not limited to):

- deployment inside industrial systems,
- predictive maintenance platforms,
- commercial software products,
- SaaS applications,
- consulting projects,
- internal enterprise tools,
- client projects,
- proprietary software integration,
- production environments,
- commercial research and development.

Commercial agreements may include:

- licensing fees,
- revenue sharing,
- consulting,
- technical collaboration,
- long-term maintenance,
- or other mutually agreed arrangements.

Commercial licensing inquiries should be directed to:

**Alireza Kafi**

through GitHub or the contact information provided in this repository.

---

# Academic Use

Academic institutions, students, educators and researchers are welcome
to use this repository for:

- learning,
- teaching,
- coursework,
- university assignments,
- master's and PhD research,
- scientific publications,
- benchmarking,
- literature comparison.

Academic citation is highly appreciated.

---

# Disclaimer

This repository is intended for research and educational purposes.

Although considerable effort has been made to build a reproducible,
scientifically motivated machine learning pipeline, this project has
**not** been validated for operational aviation use.

The implementation should **not** be used directly for:

- aircraft maintenance scheduling,
- safety-critical decision making,
- airworthiness assessment,
- certified predictive maintenance,
- operational fleet management,
- or any mission-critical environment.

No warranty is provided regarding:

- correctness,
- completeness,
- reliability,
- fitness for a particular purpose,
- regulatory compliance,
- or operational safety.

Users assume full responsibility for any use of this software.

---

# Acknowledgements

This project was inspired by the NASA C-MAPSS turbofan engine
degradation simulation dataset and was developed as a research-oriented
machine learning project focusing on predictive maintenance, Remaining
Useful Life estimation, reproducible experimentation, and leakage-safe
model development.

Special thanks to the open-source scientific Python ecosystem,
including NumPy, pandas, scikit-learn, XGBoost, Matplotlib, pytest, and
the broader machine learning community.