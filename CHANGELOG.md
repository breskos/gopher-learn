# Change Log
All notable changes to this project will be documented in this file.

## [Unreleased]

Production online learning. See [docs/service.md](docs/service.md).

No breaking API changes.

### Added
- **`service`:** a production online learning service around a network.
  - `Predict` answers with the live network and parks the input under an event ID (TTL and capacity bounded); `Feedback` / `FeedbackValue` deliver the label or value later and in any order; feedback for an already labeled event is a correction.
  - Every update trains the new sample together with a mini-batch from a replay buffer addressed by event ID, with gradient clipping, on a shadow copy of the network.
  - An outlier filter (finite values, ranges, robust z-score of regression values, custom check) rejects implausible feedback before training.
  - The errors of the live predictions feed ADWIN or Page-Hinkley; a drift raises a warning, boosts the learning rate for a while and makes the replay buffer forget the old concept.
  - Every 100 updates a guardrail scores the shadow copy on a reference set or a rolling holdout set: it goes live by an atomic swap, or it is rolled back. Versions are kept in a history, can be rolled back by hand and saved to a snapshot directory.
  - `Scaler` and `TargetScaler`: raw features and values in, raw values out.
  - Optional ε-greedy exploration for classification.
  - `Predict` never waits for training: 0.4 µs including parking; one update takes 5.7 µs for a small network.
- **Example [online-service](examples/online-service):** trains on the history of an event log, replays the rest with delayed, wrong and corrupt feedback, corrections and a drift (live 89.9 % vs. frozen 63.7 % accuracy after the drift), and serves the model over HTTP.
- **Synthetic data:** the event logs `service/offers.csv` and `service/deliveries.csv` with feedback delays, wrong and corrupt feedback, corrections and a concept drift; regression data sets can now be in raw units (`RawTarget`).

## [0.9.0] 2026-11-26

Classic models: fast, exact, deterministic baselines. See [docs/classic.md](docs/classic.md).

No breaking API changes.

### Added
- **`linear`:** `FitRegression` (least squares with optional ridge, solved exactly by Cholesky) and `FitLogistic` (multinomial logistic regression with L2, minimized with L-BFGS). Both standardize internally and report coefficients in original units; `ProbabilitiesInto` predicts without allocating.
- **`cluster`:** k-means with k-means++ starts, restarts and parallel, machine-independent assignment; `Silhouette` to choose the number of clusters.
- **`pca`:** principal component analysis from the eigen decomposition of the covariance matrix (Householder and implicit QL) with standardization, whitening, a minimum explained variance, `Inverse` reconstruction and `TransformSet`.
- `persist.SaveRegression`, `SaveLogistic`, `SaveKMeans`, `SavePCA` and their readers, validated and fuzz-tested.
- Quality tests on the synthetic data: logistic regression 100 % on blobs and macro AUC 0.992 on imbalanced classes, linear regression R² 0.72 on Friedman #1 (network 0.94), k-means 100 % purity, PCA keeps the accuracy of 200 features with 20 components.

## [0.8.0] 2026-11-26

Model selection and explainability. See [docs/selection.md](docs/selection.md) and [docs/explain.md](docs/explain.md).

No breaking API changes.

### Added
- **Hyperparameter search:** the new `tune` package. `tune.Space` lists the values to try (hidden layers, activation, learning rate, batch size, dropout, weight decay, layer normalization); `Grid` returns every combination, `Random` a sample. `tune.Search` evaluates candidates with k-fold cross-validation on the same folds for all, fits the scaler on the training part of every fold, trains with `nn.Fit` and early stopping and runs trials in parallel with results independent of the number of workers. `tune.Fit` trains the chosen candidate on all data.
- **Permutation feature importance:** the new `explain` package. `explain.Permutation` measures for a network on a data set how much worse it gets when each feature is shuffled; `model.Model.Importance` does the same for raw record columns; `explain.Permute` and `explain.PermuteParallel` work with any score function. Features are evaluated in parallel (`Config.Workers`) with results independent of the number of workers: 0.44 s instead of 2.0 s for 200 features on 8 cores. See [docs/explain.md](docs/explain.md).
- `evaluation.Metric` with `Accuracy`, `MacroF1`, `BalancedAccuracy`, `MacroAUC`, `LogLoss`, `RMSE`, `MAE` and `R2`, and `evaluation.MetricByName`.
- Quality tests: the search finds a configuration with 99.8 % on the synthetic spirals; the importance ranks the five relevant features of Friedman #1 above the five irrelevant ones.

## [0.7.0] 2026-11-25

Performance and robustness: faster on large networks, as fast on tiny ones, no allocations at inference, float32 models for small devices and protection against bad data. See [docs/performance.md](docs/performance.md) and [docs/embedded.md](docs/embedded.md).

No breaking API changes. Files written by `persist` keep format version 2.

### Performance (Apple M1, against v0.6)
| | v0.6 | v0.7 |
| --- | ---: | ---: |
| Predict, 60 × 100 × 2 | 5.2 µs | 2.6 µs |
| Predict, 784 × 128 × 64 × 10 | 128 µs | 31 µs |
| Training epoch, 784 × 128 × 64 × 10, 8 cores | 193 ms | 37 ms |
| Training epoch, same, 1 core | 193 ms | 69 ms |
| Training epoch, 60 × 100 × 2 | 9.4 ms | 5.0 ms |

### Added
- **Parallel mini-batch training:** large layers are computed on several goroutines with results that do not depend on the number of workers. `nn.Step.Workers` and `nn.FitConfig.Workers` (zero: GOMAXPROCS, one: sequential).
- **Allocation-free inference:** `Network.NewPredictor` returns a `Predictor` whose `Predict` and `Classify` do not allocate. `Network.Predict` allocates only the result, `Network.Classify` nothing.
- **float32 models:** `Network.Float32` returns an inference-only `nn.Network32` with half the memory and its own allocation-free `Predictor32`. `MarshalBinary`/`UnmarshalBinary` use a compact binary format that checks every size before allocating.
- **Regularization:** `nn.Config.Dropout` (inverted dropout on hidden layers) and `nn.Config.LayerNorm` (layer normalization with learned gain and shift); `nn.Config.Init` selects `GlorotUniform`, `GlorotNormal`, `HeUniform` or `HeNormal`. `model.Config` passes them through.
- **Gradient clipping:** `nn.Step.ClipNorm` and `nn.FitConfig.ClipNorm`.
- `nn.ErrNonFinite`: a training step with a NaN or infinite loss or gradient is skipped instead of destroying the weights; `nn.Fit` stops with `StopDiverged` and keeps the last finite weights.
- `learn.Set.Validate`; `persist.MaxDocumentSize` and `persist.ErrTooLarge` limit the size of loaded documents.
- Benchmarks for `nn`, `model`, `online` and `drift`; CI compares them with the base branch on every pull request (`internal/cmd/benchcheck`) and fails on regressions. CI also tests on 32-bit and builds for ARM, RISC-V and WebAssembly.
- Fuzz tests for every reader in `persist` and for the float32 binary format.
- `examples/embedded`: train a motor fault detector on a computer and run it as a 1 KB float32 model on a device.

### Changed
- `persist.ReadSet` and `ReadOnline` validate what they load (sizes, finite values, classes, configuration), so corrupt files fail on load instead of panicking later.
- The online learner skips training on samples with a non-finite gradient.
- `drift.ADWIN` no longer allocates when merging buckets.
- `persist` flushes files to the disk before renaming them, so a power loss leaves either the old or the new file.

## [0.6.0] 2026-11-25

Online learning done right: test-then-train evaluation, concept drift detection with ADWIN and Page-Hinkley, bounded memory and encoders for text streams.

No breaking API changes. `online.DefaultConfig()` now bounds the memory and detects drift; see Changed.

### Added
- **Drift detection:** the new `drift` package with `drift.ADWIN` (adaptive windowing; `Change` tells whether the values rose or fell) and `drift.PageHinkley`, both behind the `drift.Detector` interface and serializable.
- **Test-then-train:** `online.Online.Learn(sample)` predicts a sample, records the result, feeds the error to the drift detector, remembers the sample and trains on it. It returns an `online.Result` with the prediction, the error and whether drift was detected.
- **Prequential evaluation:** `evaluation.Prequential` with windowed accuracy, MAE and RMSE plus a cumulative `Evaluation`; the learner keeps one in `Online.Prequential`.
- **Bounded memory:** `Config.Memory` is `online.Window` (most recent samples), `online.Reservoir` (uniform sample of the stream) or `online.Unbounded`, with `Config.MemorySize`.
- **Drift reaction:** on a rising error the learner forgets the samples from before the change; `Config.OnDrift` is called with an `online.DriftEvent`, at most once per `Config.DriftCooldown` samples. `Config.Drift` selects ADWIN, Page-Hinkley or none.
- **Encoders:** `encoders.StringHashing` (hashing trick, fixed size, no vocabulary), `encoders.StringTFIDF` and `encoders.FloatCategories` (one-hot for numeric category codes), with `Config.HashDimensions`, `TFIDFMaxTerms` and `CategoriesMax`.
- `learn.Set.Prepare`, `Replace` and `DropOldest`.
- Quality tests on the synthetic drift streams: for classification (abrupt, gradual, recurring) drift is detected right after the change and the accuracy after an abrupt drift reaches 0.92 instead of 0.72 without detection; for regression the window memory and drift detection lower the prequential MAE on slowly drifting coefficients and after the replacement of an aging air quality sensor.

### Changed
- `online.DefaultConfig()` now bounds the memory to a window of 5,000 samples and detects drift with ADWIN. Set `Memory = online.Unbounded` and `Drift = online.NoDrift` for the previous behavior. Learners saved with earlier versions load with unbounded memory and no drift detection.
- `Inject` respects the memory bounds.
- `Online.Iterate` trains with `nn.Fit` and keeps the weights of the epoch with the lowest loss on the held-out split. On the synthetic streams this lowers the prequential error by up to 40 % (e.g. 0.024 → 0.015 on sensor health, 0.125 → 0.115 on recurring concepts). The new `Config.Optimizer` selects the optimizer; the default stays SGD, which did better than Adam here.

## [0.5.0] 2026-11-25

Tabular essentials: train on raw CSV records, scale and split data properly, train with Adam and early stopping, measure with the usual metrics and ship the whole pipeline as one file.

No breaking API changes. Note that files are now written in persistence format version 2, which gopher-learn 0.4 cannot read; 0.5 reads both versions.

### Added
- **CSV options** for `learn.Set.ReadCSV` and `LoadCSV`: `WithHeader`, `WithTarget(name)`, `WithTargetColumn(index)`, `WithIgnore(names...)`, `WithIgnoreColumns(indices...)` and `WithComma(rune)`. Without options the behavior is unchanged. Header names are stored in the new `Set.Features` field.
- **Feature scaling:** `learn.FitScaler` with `learn.Standardize` (zero mean, unit variance) and `learn.MinMax` (range [0, 1]). A `Scaler` transforms vectors and whole sets, can invert the transformation and serializes to JSON.
- **Stratified split:** `learn.Set.StratifiedSplit` keeps the class proportions in both parts.
- **Cross-validation:** `learn.Set.KFold` returns k stratified train/test folds.
- **Optimizers:** `nn.SGD`, `nn.Momentum` and `nn.Adam` (`nn.DefaultAdam`) behind the `nn.Optimizer` interface, plus decoupled L2 weight decay. `Network.TrainBatchWith` applies one step with any optimizer; `TrainBatch` keeps plain SGD.
- **Training API:** `nn.Fit` trains on `nn.Data` (`learn.Set.Data` converts a set) with mini-batches, a learning-rate schedule (`nn.StepDecay`, `nn.CosineDecay`), early stopping on the validation loss that restores the best weights, cancellation via context and an `OnEpoch` callback. It returns a `History` of every epoch and why training stopped. With Adam, configurations that got stuck with SGD (e.g. tanh on spirals) now train reliably.
- `Network.Loss` computes the mean loss on samples without training.
- **Metrics:** MAE, RMSE and R² for regression; log loss, per-class ROC-AUC and macro AUC for classification. The reports include them.
- **Models:** the new `model` package goes from raw records to predictions. `model.Train` detects numeric and categorical columns, fits encoders and a scaler on the training part only, standardizes regression targets and trains a network with `nn.Fit` and early stopping. `Model.Predict(record)` takes raw values and returns the label with scores or the value in the units of the target; `Model.Evaluate` scores records with targets. `model.ReadRecords` and `LoadRecords` read CSV files with a header row. See [docs/model.md](docs/model.md).
- **Persistence format version 2:** `persist.SaveModel`/`LoadModel` store a model with its encoders, scaler and network as one file and validate it on load; `persist.SaveScaler`/`LoadScaler` store scalers. Files are now written with version 2; version 1 files stay readable.
- `learn.FitScalerVectors` fits a scaler on plain vectors.
- **Synthetic data sets** for testing and benchmarks (`internal/synth`), grouped by purpose: `classification` and `regression` for batch training, `online` for streams, `text` for the encoders, `loading` for loader edge cases and `benchmark` for performance. They include mathematical benchmarks (blobs, spirals, Friedman #1, ...), applied sensor data generated from physical and behavioral models (predictive maintenance, run-to-failure, activity recognition, Wi-Fi localization, air quality calibration, smart meter, room occupancy, sensor health) and streams for online learning with abrupt, gradual, recurring and incremental drift plus an aging air quality sensor. Streams carry the true concept per row to evaluate drift detection.
- `go run ./internal/cmd/gendata -out <dir>` writes the data with a README and a small Go module (standard library only) containing a catalog, a reader for every format, examples and a test that checks every file. The generated files are published in [gopher-learn-data](https://github.com/breskos/gopher-learn-data/tree/main/synthetic).
- **Quality tests** train models on the synthetic data and fail if the accuracy on a held-out test set drops below fixed thresholds. Online tests feed the streams row by row to `online.Online` and check that each stream is learnable before its drift and that the drift is real. `go test -short` skips them.

### Changed
- The `engine` trains on `nn.Fit`: each try is one run whose schedule lowers the learning rate after every round and whose epoch callback evaluates the validation split at the end of a round. New `engine.Config` fields: `Optimizer` (a function returning a fresh optimizer per try; default SGD as before) and `WeightDecay`. A diverging try ends early with a warning instead of training on with NaN weights.
- `engine` and `online` split classification data stratified, so rare classes are always present in training and validation.
- Empty CSV fields are reported as missing values, naming the column.
- The wine-quality example holds back a test set and standardizes the features.

## [0.4.1] 2026-11-25

Project hygiene and robustness. No breaking API changes.

### Fixed
- `learn.Set.ReadSVM` allocated a dense vector sized by the largest feature index for every row, so a single line such as `1 2147483647:1` tried to allocate about 16 GB. Feature indices above 1,048,576 and inputs with more than 134,217,728 values are now rejected with an error.
- `learn.Set.Add` accepted `NaN` and infinite values (also from CSV and LIBSVM files) and samples without features. They are now rejected; `learn.ErrNonFinite` identifies non-finite values.
- `online.Online.Inject` added a sample with the wrong number of features to the data set before failing, after which every correct sample was rejected. The dimension is now checked first.

### Added
- Fuzz tests for the CSV and LIBSVM loaders, run for 30 seconds each in CI.
- Tests for every package; total coverage is 97.9 %. CI fails below 90 %.
- `CONTRIBUTING.md`, issue and pull request templates, Go Report Card badge.

## [0.4.0] 2026-11-25

Complete rewrite for current Go. This release breaks the API; see the migration notes below.

### Changed
- License changed from GPL-3.0 to Apache-2.0.
- Go module (`go.mod`), requires Go 1.24.
- New package `nn` replaces `net` (which shadowed the standard library). The network is now based on dense weight matrices instead of neuron and synapse objects, has bias terms, and trains about 1.6× and predicts about 1.9× faster (60-100-2 network, allocation-free training).
- Library code never exits the program or panics on bad input anymore; all fallible functions return errors.
- Training is reproducible with `Config.Seed` and cancellable via `context.Context`.
- Logging via `log/slog` (`Config.Logger`) instead of printing to stdout.
- Reports are written to an `io.Writer`; output order is deterministic.
- Persistence uses a versioned JSON envelope, writes files atomically and validates loaded networks. Files written by 0.3 cannot be loaded.
- `learn.Set.Split` shuffles and splits exactly by ratio; each engine try starts with a freshly initialized network.

### Added
- Activations: tanh, ReLU, linear, softmax (with cross-entropy loss).
- Mini-batch training (`Config.BatchSize`, `nn.Network.TrainBatch`).
- `io.Reader` and `io.Writer` variants for loading data and persisting models.
- GitHub Actions CI with race detector and golangci-lint, Dependabot.
- Tests for every package, including a numerical gradient check.

### Fixed
- `FloatReducer` produced dimensions in random order (map iteration).
- `StringDictionary` normalized strings when fitting but not when encoding.
- `StringSplitDictionary` looked up the whole string instead of its tokens, and normalization removed the spaces it split on.
- String normalization dropped non-ASCII letters such as umlauts.
- N-gram vector indices depended on map iteration order.
- `FalseNegatives` was computed incorrectly.
- Loading a CSV left samples with too short target vectors when a new class appeared later in the file.
- LIBSVM loader dropped the highest feature index and did not build target vectors.
- `online.Init` stopped as soon as the score was *below* the target.
- Persisting an online learner stored the output size as input size.
- Metrics returned NaN on empty evaluations.

### Migration from 0.3

| 0.3 | 0.4 |
| --- | --- |
| `net.NewNetwork`, `net.BuildNetwork` | `nn.New(nn.Config{...})` |
| `net.Classification`, `net.Regression` | `nn.Classification`, `nn.Regression` |
| `net.Distance` and other criteria | `engine.Distance`, ... |
| `network.Calculate(v)` | `network.Predict(v)` (returns an error) |
| `network.CalculateWinnerLabel(v)` | `network.Classify(v)` |
| `network.CalculateLabels(v)` | `network.Scores(v)` |
| `learn.Learn(n, in, out, rate)` | `network.Train(in, out, rate)` |
| `set.LoadFromCSV(path)` | `set.LoadCSV(path)` / `set.ReadCSV(r)` |
| `set.LoadFromSVMFile(path)` | `set.LoadSVM(path)` / `set.ReadSVM(r)` |
| `set.AddSample(s)` | `set.Add(s)` |
| `set.ClassToLabel` | `set.Labels` |
| `set.GenerateOutputVector(label)` | `set.OneHot(label)` (target vectors are filled in by `Add`) |
| `learn.NewClassificationSample(v, out, label)` | `learn.NewClassificationSample(v, label)` |
| `engine.NewEngine(usage, hidden, data)` + `SetConfig` + `Start(c)` | `engine.New(data, hidden, cfg)` + `Run(ctx, c)` |
| `e.GetWinner()` | `e.Winner()` |
| `online.NewOnline(usage, inputs, hidden, set)` | `online.New(set, inputs, hidden, cfg)` |
| `o.Iterate()` | `o.Iterate(ctx)` (returns `ErrNotEnoughData` early on) |
| `SetVerbose(true)` | `Config.Logger` |
| `evaluation.GetX(...)` | `evaluation.X(...)` |
| `PrintSummary(label)` etc. | `WriteSummary(w, label)` etc. |
| `persist.ToFile` / `FromFile` | `persist.SaveNetwork` / `LoadNetwork` |
| `persist.SetToFile`, `OnlineToFile`, `EncoderToFile` | `persist.SaveSet`, `SaveOnline`, `SaveEncoder` (and `Load*`) |
| `encoders.NewEncoder(name)` | `encoders.New(name)` |
| `encoders.NewInput(name, type)` | `encoders.NewInput(type)` |
| `e.Transform(name, input)` | `e.Fit(name, input)` |
| `e.Encode(name, encoders.Unified{...})` | `e.Encode(name, encoders.Value{...})`, `EncodeString`, `EncodeFloats` |
| `encoders.EncoderConfig` | `encoders.Config` |

## [0.3.0] 2021-06-07

This version introduces a persitence layer for encoders.

### Added
- Serialization for encoders
  
### Changed
- Interface of encoders
- Minor things in the library

### Fixed
- Some issues with serialization of online, network


## [0.2.5] 2021-06-06

With this version we introduce encoders (automatic encoders) to gopher-learn.
You now can reduce large float slice inputs or encode your string input right away.

### Added
- Encoders for float slices and string input.
- With encoders large float input can be reduced using Spearman.
- Also with encoders strings can be encoded as ngrams and dictionary (Topic modelling to come soon)
  
### Changed
- Relocated the neural net from neural package into an own package called net

### Fixed
- Nothing here


## [0.2] - 2021-05-09
 
Introducing online learning.
 
### Added
- Config for online learner to control learning behavior - easily inject your own config
- Config for engine learner to control learning behavior - easily inject your own config
- Wrote Comments to every function to make everything easier to understand
- Online learner can now be serialized to disk using persist.OnlineToFile() and load using persist.OnlineFromFile()
  
### Changed
- Refactoring of Criterion handling. It is now part of the neural package (usage e.g.: neural.Distance)
- Reworked some of the examples

### Fixed
- Fixed regression example, did not work correctly
