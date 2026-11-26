# Roadmap

> **The simplest pure-Go library for tabular data and online learning — no cgo, embeddable in any Go service.**

gopher-learn does not try to compete with PyTorch, GoMLX or ONNX Runtime. It focuses on small and medium tabular models and on models that keep learning while a service runs. Every feature below is judged against that focus.

Until v1.0.0 minor versions may change the API; every breaking change is listed in the [changelog](CHANGELOG.md) with migration notes. Patch versions never break the API.

## Overview

| Version | Theme | Status |
| --- | --- | --- |
| v0.4.0 | Rewrite for current Go, Apache-2.0 | released |
| v0.4.x | Project hygiene | released (v0.4.1) |
| v0.5.0 | Tabular essentials | released |
| v0.6.0 | Online learning done right | released |
| v0.7.0 | Performance and robustness | released |
| v0.8.0 | Model selection and explainability | released |
| v0.9.0 | Classic models | released |
| v0.10.0 | Production online learning | in progress |
| v0.11.0 | Gradient-boosted trees | planned |
| v0.12.0 | Interop and tooling | planned |
| v0.13.0 | WebAssembly and TinyGo | planned |
| v0.14.0 | Developer experience | planned, after the technical work |
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

Goal: fast, small enough for embedded devices and robust against bad data.

- Parallel mini-batch training with goroutines
- Optional `float32` networks (inference-only `Network32` with a compact binary format)
- Benchmarks in CI with regression alerts
- Allocation-free inference, faster pure-Go kernels and hardened loading
- Dropout, layer normalization, selectable weight initialization
- ~~Optional BLAS backend behind a build tag~~: dropped, because it needs cgo, which defeats the embedded goal; optimized pure-Go kernels instead

## v0.8.0 — Model selection and explainability

- Hyperparameter search (random and grid) on top of `nn.Fit` and cross-validation
- Permutation feature importance

## v0.9.0 — Classic models

Strong, fast baselines that every tabular project needs.

- Linear and logistic regression
- k-means clustering
- PCA

## v0.10.0 — Production online learning

Goal: a stable online learning service that works with delayed feedback and later corrections of labels and values, and protects the live model against noisy data and degradation.

Implemented in the [`service`](service) package, see [docs/service.md](docs/service.md).

Data flow: input → `event_id` and live prediction → park the input vector (ID store) → later feedback or correction by `event_id` → outlier filter and replay buffer → online update (with clipping and drift check) → guardrail test → live swap or rollback.

### Functional requirements

| ID | Requirement | Description | Status |
| --- | --- | --- | --- |
| FR-01 | ID-based delayed feedback | Every prediction gets a unique `event_id`. Its input vector is stored temporarily until the true label or regression value arrives. | done: `Predict` / `PredictID` and `Feedback`, with TTL and capacity |
| FR-02 | Later correction | Labels or values already given can be overwritten by `event_id`, which triggers another update. | done: feedback for a labeled event replaces it and trains again |
| FR-03 | Experience replay | A sliding window (ring buffer) of past labeled samples, addressable by `event_id`. Every online update trains the new or corrected sample together with a sample from the buffer to prevent catastrophic forgetting. | done: ring buffer by event ID, 15 replay samples per update |
| FR-04 | Concept drift detection | Continuous monitoring of the prediction error. On significant changes the learning rate is adapted automatically or a warning is raised. | done: ADWIN / Page-Hinkley, `OnDrift` warning, learning-rate boost, replay forgets the old concept |
| FR-05 | Exploration vs. exploitation (optional, for agents) | Controlled trying of alternative actions (e.g. ε-greedy) to avoid dead ends caused by one-sided feedback. | done: `Epsilon` for classification |

### Non-functional requirements and guardrails

| ID | Requirement | Description | Status |
| --- | --- | --- | --- |
| NFR-01 | Outlier filter and plausibility | Incoming feedback is checked for outliers and valid ranges before training, so single corrupt values cannot destroy the network. | done: ranges, robust z-score, custom `Check`; regression needs no targets in [0, 1] (linear outputs and target scaling exist since v0.5) |
| NFR-02 | Gradient clipping / step limit | The largest weight change per update is limited. | done: `ClipNorm` on every update, non-finite steps skipped |
| NFR-03 | Snapshot versioning and auto-rollback | Model weights are persisted regularly. Before going live (or periodically) the network is checked against a stable reference set; if the quality drops, it is rolled back automatically. | done: versions with history, reference set or rolling holdout, auto-rollback, `SnapshotDir` |
| NFR-04 | Decoupled inference and training | Live predictions must not be blocked or see inconsistent weights while updates run (shadow update and atomic model swap). | done: live version behind an atomic pointer, training on a shadow copy |

## v0.11.0 — Gradient-boosted trees

The state of the art for tabular data and usually stronger than neural networks there.

- Gradient-boosted decision trees for classification and regression
- Same `Fit` / `Predict` / persistence interface as the other models

## v0.12.0 — Interop and tooling

- **ONNX export** to serve gopher-learn models anywhere and to exchange models with the Python ecosystem
- **CLI:** `gopher-learn train data.csv` and `gopher-learn evaluate`
- **Serving:** `gopher-learn serve model.json` as an HTTP endpoint

## v0.13.0 — WebAssembly and TinyGo

- Inference in the browser (WebAssembly) and on edge devices (TinyGo)

## v0.14.0 — Developer experience

Make it easy and fun to explore what gopher-learn can do. This comes after the technical releases; the training callbacks it builds on (`nn.FitConfig.OnEpoch`) are already part of v0.5.

- **Training doctor:** warnings with hints for typical mistakes: stagnating or diverging loss, saturated neurons (learning rate too high), unscaled features, strongly imbalanced classes misleading accuracy, NaN values
- **Model summary:** `network.Summary()` with layers, activations and parameter count
- **HTML report:** one call writes `report.html` with loss curves, confusion matrix, per-class metrics and the network summary (standard library and inline SVG only)
- **Online learning simulator:** replay a stream such as `online/abrupt-drift.csv`, plot accuracy over time, mark the true drift points from the `concept` column and compare strategies
- **CLI for exploring:** `gopher-learn datasets list`, `gopher-learn try <data set>` with the synthetic data generated on the fly, plus `train` and `predict` (extends the v0.12 CLI)
- **Browser playground:** gopher-learn compiled to WebAssembly; pick a data set, set layers and learning rate and watch the decision boundary form live (builds on v0.13)
- **Go notebooks:** interactive Jupyter notebooks in Go (GoNB), one per data set folder
- **Tour of gopher-learn:** a guided step-by-step introduction like the Go tour: load, scale, train, evaluate, learn online, persist
- **Service example:** a small HTTP service with `/predict` and `/learn` that keeps learning and survives restarts
- **Reference results:** the expected quality per synthetic data set in the data catalog, so users know whether their setup does well

## v1.0.0 — Stable API

- API freeze with semantic-versioning guarantees
- Test coverage above 85 % in every package
- Complete documentation and guides

## Not planned

- **Automatic differentiation and deep-learning layers** (convolution, recurrent, attention). These are better served by GoMLX or ONNX Runtime and would pull gopher-learn away from its focus.
