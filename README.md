# gopher-learn

[![Go Reference](https://pkg.go.dev/badge/github.com/breskos/gopher-learn.svg)](https://pkg.go.dev/github.com/breskos/gopher-learn)
[![CI](https://github.com/breskos/gopher-learn/actions/workflows/ci.yml/badge.svg)](https://github.com/breskos/gopher-learn/actions/workflows/ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/breskos/gopher-learn)](https://goreportcard.com/report/github.com/breskos/gopher-learn)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

![The gopher-learn logo](https://breskos.com/wp-content/uploads/2026/11/gopher.png)

Neural networks in pure Go, with no dependencies outside the standard library. Fast enough for services and small enough for embedded devices.

- Models that go from raw CSV records to predictions: column detection, encoders, scaling and the network saved as one file
- Feed-forward networks (multi-layer perceptrons) with sigmoid, tanh, ReLU, linear and softmax activations, SGD, momentum and Adam, dropout, layer normalization and early stopping
- Fast: batched training on all CPU cores with bit-identical results, allocation-free inference, and float32 models with a compact binary format for small devices
- A training engine that decays the learning rate and keeps the best network according to a criterion of your choice
- Online learning for streams: test-then-train evaluation, bounded memory and concept drift detection (ADWIN, Page-Hinkley)
- Encoders that turn strings and float vectors into feature vectors, including the hashing trick and TF-IDF for text streams
- Rich evaluation: confusion matrix, precision, recall, F1, balanced accuracy, ROC-AUC, log loss, RMSE, R² and more
- Classic models as fast, exact baselines: linear and logistic regression, k-means clustering and PCA
- Model selection and explainability: grid and random hyperparameter search with cross-validation, permutation feature importance
- Versioned JSON persistence for models, networks, data sets, online learners, encoders and scalers

## Performance

Built for ordinary CPUs and small devices: no cgo, no BLAS, no GPU. Measured on an Apple M1 ([details](docs/performance.md)):

| Network | Prediction | Training epoch (1,024 samples) |
| --- | ---: | ---: |
| tiny sensor model, 4 × 8 × 2 | 0.12 µs | 0.22 ms |
| sonar, 60 × 100 × 2 | 2.6 µs | 5.0 ms |
| MNIST-sized, 784 × 128 × 64 × 10 | 31 µs | 37 ms |

- **Fast kernels:** dot products with eight accumulators and fused multiply-add, cache-friendly batched training; 2 to 5 times faster than v0.6 on medium and large networks.
- **All cores, same result:** large layers are trained in parallel, and the trained network is bit for bit identical for any number of cores.
- **No garbage:** prediction with a `Predictor` and every training step allocate no memory.
- **Robust:** a NaN or infinite gradient never destroys a network, every loader validates what it reads, and all readers are fuzz-tested.
- **Checked in CI:** every pull request is benchmarked against the base branch and fails on regressions.

## Embedded systems

Train on a computer, predict on the device ([guide](docs/embedded.md), [example](examples/embedded)):

```go
data, _ := network.Float32().MarshalBinary() // on the computer: 4 bytes per parameter

var n nn.Network32                           // on the device: only the nn package
if err := n.UnmarshalBinary(data); err != nil {
	return err
}
label, err := n.NewPredictor().Classify(reading) // no allocation
```

A motor fault detector with 234 parameters is a 1 KB model file; the complete device program is 1.5 to 1.9 MB and classifies 1.5 million readings per second on one core of an M1. CI builds for ARMv7, RISC-V and WebAssembly and runs the tests on 32-bit.

## Explainability

Which inputs does a model rely on? `explain.Permutation` shuffles one feature at a time and measures how much worse the model gets ([guide](docs/explain.md)):

```go
imp, _ := m.Importance(testRecords, explain.Config{}) // or explain.Permutation(network, testSet, …)
for _, f := range imp {
	fmt.Printf("%-8s %.3f ± %.3f\n", f.Feature, f.Mean, f.Std)
}
```

```
income   0.326 ± 0.021   ← 33 accuracy points lost without it
x        0.243 ± 0.015
color    0.186 ± 0.009
noise    0.002 ± 0.003   ← unused
```

It works for every model, needs no retraining, runs on all cores and exposes data leaks: a column that is only known after the outcome stands out with a far higher importance than everything else.

## Install

```
go get github.com/breskos/gopher-learn@latest
```

Requires Go 1.24 or newer.

## Packages

| Package | Purpose |
| --- | --- |
| [`model`](model) | Raw records in, predictions out: preprocessing and network as one artifact |
| [`nn`](nn) | The neural network: create, train with `nn.Fit`, predict |
| [`learn`](learn) | Data sets and samples, CSV and LIBSVM loaders |
| [`engine`](engine) | Batch training with learning rate decay and model selection |
| [`online`](online) | Continuous training from a stream of samples |
| [`drift`](drift) | Concept drift detection: ADWIN and Page-Hinkley |
| [`evaluation`](evaluation) | Metrics and reports |
| [`linear`](linear) | Linear (ridge) and logistic regression |
| [`cluster`](cluster) | k-means clustering |
| [`pca`](pca) | Principal component analysis |
| [`tune`](tune) | Hyperparameter search with cross-validation |
| [`explain`](explain) | Permutation feature importance |
| [`encoders`](encoders) | Feature encoding for strings and float vectors |
| [`persist`](persist) | Save and load everything as JSON |

## Quickstart

Train a model on a CSV file with a header row and predict from raw values:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/breskos/gopher-learn/model"
	"github.com/breskos/gopher-learn/persist"
)

func main() {
	records, err := model.LoadRecords("machines.csv")
	if err != nil {
		log.Fatal(err)
	}
	m, _, err := model.Train(context.Background(), records, model.Config{
		Target: "status",
		Ignore: []string{"machine_id"},
	})
	if err != nil {
		log.Fatal(err)
	}
	if err := persist.SaveModel("model.json", m); err != nil {
		log.Fatal(err)
	}

	p, err := m.Predict(model.Record{"temperature": "91.5", "vibration": "2.1", "vendor": "acme"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(p.Label, p.Scores)
}
```

Numeric columns are scaled, text columns are encoded, and the network trains with Adam and early stopping. Use `model.Config{Task: nn.Regression}` to predict numbers. See [Models](docs/model.md).

### Using the engine

Train a classifier from a CSV file whose last column is the class label:

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/breskos/gopher-learn/engine"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/persist"
)

func main() {
	data := learn.NewSet(nn.Classification)
	if err := data.LoadCSV("data.csv"); err != nil {
		log.Fatal(err)
	}

	e, err := engine.New(data, []int{20}, engine.DefaultConfig())
	if err != nil {
		log.Fatal(err)
	}
	if err := e.Run(context.Background(), engine.FMeasure); err != nil {
		log.Fatal(err)
	}

	network, eval := e.Winner()
	_ = eval.WriteSummaries(os.Stdout)
	_ = eval.WriteConfusionMatrix(os.Stdout)

	if err := persist.SaveNetwork("network.json", network); err != nil {
		log.Fatal(err)
	}

	label, err := network.Classify(data.Samples[0].Vector)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("%s -> %s", data.Samples[0].Label, label)
}
```

For regression use `learn.NewSet(nn.Regression)`, choose `engine.Distance` as criterion and call `network.Predict(vector)`. With the default sigmoid output layer, targets must lie in [0, 1]; set `Config.OutputActivation = nn.Linear` for unbounded targets.

### Using the network directly

```go
net, err := nn.New(nn.Config{
	Inputs:           2,
	Hidden:           []int{8},
	Labels:           []string{"same", "different"},
	HiddenActivation: nn.Tanh,
	OutputActivation: nn.Softmax,
})
// ...
loss, err := net.TrainBatch(inputs, targets, 0.5)
label, err := net.Classify([]float64{0, 1})
```

## Documentation

- [Models](docs/model.md)
- [Classic models](docs/classic.md)
- [Model selection](docs/selection.md)
- [Explainability](docs/explain.md)
- [Performance](docs/performance.md)
- [Embedded systems](docs/embedded.md)
- [Engine](docs/engine.md)
- [Online learning](docs/online.md)
- [Evaluation](docs/evaluation.md)
- [Encoders](docs/encoders.md)
- [Examples](examples), with data in [gopher-learn-data](https://github.com/breskos/gopher-learn-data)
- [Changelog](CHANGELOG.md), including the migration guide from 0.3
- [Roadmap](ROADMAP.md)
- [Contributing](CONTRIBUTING.md)

## License

Apache License 2.0, see [LICENSE](LICENSE). Versions up to and including v0.3.0 were released under the GPL-3.0.
