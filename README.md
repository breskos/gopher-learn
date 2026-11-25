# gopher-learn

[![Go Reference](https://pkg.go.dev/badge/github.com/breskos/gopher-learn.svg)](https://pkg.go.dev/github.com/breskos/gopher-learn)
[![CI](https://github.com/breskos/gopher-learn/actions/workflows/ci.yml/badge.svg)](https://github.com/breskos/gopher-learn/actions/workflows/ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/breskos/gopher-learn)](https://goreportcard.com/report/github.com/breskos/gopher-learn)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

![The gopher-learn logo](https://breskos.com/wp-content/uploads/2026/11/gopher.png)

Neural networks in pure Go, with no dependencies outside the standard library.

- Models that go from raw CSV records to predictions: column detection, encoders, scaling and the network saved as one file
- Feed-forward networks (multi-layer perceptrons) on dense matrices with biases, sigmoid, tanh, ReLU, linear and softmax activations, and mini-batch gradient descent
- A training engine that decays the learning rate and keeps the best network according to a criterion of your choice
- Online learning: inject samples one by one, for streams and autonomous agents
- Encoders that turn strings and large float vectors into feature vectors
- Rich evaluation: confusion matrix, precision, recall, F1, balanced accuracy and more
- Versioned JSON persistence for models, networks, data sets, online learners, encoders and scalers

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
| [`evaluation`](evaluation) | Metrics and reports |
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
