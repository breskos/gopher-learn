# Roadmap

> **The simplest pure-Go library for tabular data and online learning — no cgo, embeddable in any Go service.**

gopher-learn does not try to compete with PyTorch, GoMLX or ONNX Runtime. It focuses on small and medium tabular models and on models that keep learning while a service runs. Every feature below is judged against that focus.

Until v1.0.0 minor versions may change the API; every breaking change is listed in the [changelog](CHANGELOG.md) with migration notes. Patch versions never break the API.

## Overview

| Version | Theme | Status |
| --- | --- | --- |
| v0.4.0 | Rewrite for current Go, Apache-2.0 | released |
| v0.4.x | Project hygiene | planned |
| v0.5.0 | Tabular essentials | planned |
| v0.6.0 | Online learning done right | planned |
| v0.7.0 | Performance and robustness | planned |
| v0.8.0 | Model selection and explainability | planned |
| v0.9.0 | Classic models | planned |
| v0.10.0 | Gradient-boosted trees | planned |
| v0.11.0 | Interop and tooling | planned |
| v0.12.0 | WebAssembly and TinyGo | planned |
| v1.0.0 | Stable API | planned |

## v0.4.x — Project hygiene

No API changes.

- Runnable examples (`Example…` tests) for every package on pkg.go.dev
- Fuzz tests for the CSV and LIBSVM loaders
- `CONTRIBUTING.md`, issue and pull request templates, `good first issue` labels
- Coverage and Go Report Card badges
- Move the logo into the repository (served over HTTPS)

## v0.5.0 — Tabular essentials

Make gopher-learn a serious choice for everyday tabular problems.

- **Feature scaling:** standard and min-max scalers in `learn`, fitted on training data and persisted
- **CSV loading:** header rows, selectable target column, ignored columns, clear errors for missing values
- **Metrics:** MAE, RMSE and R² for regression; log loss and ROC-AUC for classification
- **Optimizers:** momentum and Adam, plus L2 weight decay
- **Training API:** `nn.Fit` with epochs, batch size, validation split, learning-rate schedule and early stopping; the engine builds on it
- **Data splits:** stratified split and k-fold cross-validation
- **Pipeline model:** encoders, scaler and network saved as one artifact, used with `model.Predict(record)` on raw input
- Persistence format version 2 (version 1 files stay readable)

## v0.6.0 — Online learning done right

The distinguishing feature of gopher-learn.

- **Prequential evaluation** (test-then-train) instead of evaluating on samples that were already trained on
- **Concept drift detection** (ADWIN, Page-Hinkley) with hooks to react to drift
- **Bounded memory:** sliding window and reservoir sampling instead of an ever-growing data set
- **Encoders:** hashing trick (fixed dimensions, ideal for streams), TF-IDF weighting, one-hot for numeric categories

## v0.7.0 — Performance and robustness

- Parallel mini-batch training with goroutines
- Optional `float32` networks
- Benchmarks in CI with regression alerts
- Optional BLAS backend behind a build tag (default stays pure Go)
- Dropout, layer normalization, selectable weight initialization

## v0.8.0 — Model selection and explainability

- Hyperparameter search (random and grid) on top of `nn.Fit` and cross-validation
- Permutation feature importance

## v0.9.0 — Classic models

Strong, fast baselines that every tabular project needs.

- Linear and logistic regression
- k-means clustering
- PCA

## v0.10.0 — Gradient-boosted trees

The state of the art for tabular data and usually stronger than neural networks there.

- Gradient-boosted decision trees for classification and regression
- Same `Fit` / `Predict` / persistence interface as the other models

## v0.11.0 — Interop and tooling

- **ONNX export** to serve gopher-learn models anywhere and to exchange models with the Python ecosystem
- **CLI:** `gopher-learn train data.csv` and `gopher-learn evaluate`
- **Serving:** `gopher-learn serve model.json` as an HTTP endpoint

## v0.12.0 — WebAssembly and TinyGo

- Inference in the browser (WebAssembly) and on edge devices (TinyGo)

## v1.0.0 — Stable API

- API freeze with semantic-versioning guarantees
- Test coverage above 85 % in every package
- Complete documentation and guides

## Not planned

- **Automatic differentiation and deep-learning layers** (convolution, recurrent, attention). These are better served by GoMLX or ONNX Runtime and would pull gopher-learn away from its focus.
