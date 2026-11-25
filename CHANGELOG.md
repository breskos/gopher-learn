# Change Log
All notable changes to this project will be documented in this file.

## [0.4.1] Unreleased

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
