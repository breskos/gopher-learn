# Performance

gopher-learn is built for speed on ordinary CPUs: no cgo, no BLAS, no GPU, only Go and the standard library. Large networks are 2 to 5 times faster than in v0.6, prediction allocates no memory, and training uses all CPU cores while producing exactly the same result as on one core.

For small devices see [Embedded systems](embedded.md).

## Benchmarks

Measured on an Apple M1 (8 cores, ARM64) with Go 1.27; median of six runs of `make bench`. The training epoch is 1,024 samples in mini-batches of 32 with Adam.

| Network | Parameters | Prediction | Predictions per second (1 core) | Training epoch, 8 cores | Training epoch, 1 core |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny sensor model, 4 × 8 × 2 | 58 | 0.12 µs | 8.1 million | 0.22 ms | 0.22 ms |
| sonar, 60 × 100 × 2 | 6,302 | 2.6 µs | 385,000 | 5.0 ms | 5.4 ms |
| MNIST-sized, 784 × 128 × 64 × 10 | 109,386 | 31 µs | 32,000 | 37 ms | 69 ms |

Prediction is measured with a `Predictor`, which allocates nothing. `Network.Predict` takes the same time and allocates only the result vector.

Other operations:

| Operation | Time | Allocations |
| --- | ---: | ---: |
| `model.Model.Predict` from a raw record (encoding, scaling, network) | 0.7 µs | 8 |
| `online.Online.Learn` (test, drift detection, memory, 5 training steps) | 2.4 µs | 5 |
| `drift.ADWIN.Add` | 0.28 µs | 0 |
| `drift.PageHinkley.Add` | 9 ns | 0 |
| Loading a 6,302-parameter model: float32 binary / JSON | 8.7 µs / 1.2 ms | |

### Compared with v0.6

| | v0.6 | v0.7 | |
| --- | ---: | ---: | ---: |
| Prediction, 60 × 100 × 2 | 5.2 µs | 2.6 µs | 2.0 × faster |
| Prediction, 784 × 128 × 64 × 10 | 128 µs | 31 µs | 4.1 × faster |
| Training epoch, 784 × 128 × 64 × 10 | 193 ms | 37 ms | 5.2 × faster (2.8 × on 1 core) |
| Training epoch, 60 × 100 × 2 | 9.4 ms | 5.0 ms | 1.9 × faster |
| Prediction / training epoch, 4 × 8 × 2 | 128 ns / 209 µs | 134 ns / 220 µs | about the same |
| Allocations per prediction | 2 to 3 | 0 (`Predictor`) or 1 | |

Tiny networks were already limited by the activation functions; the optimizations target everything from a few hundred parameters upwards.

## How it is optimized

**Dot products with eight accumulators.** A dense layer is a sequence of dot products. A single running sum makes every multiply-add wait for the previous one. Eight independent sums let the CPU work on eight at once, and Go compiles them to fused multiply-add instructions on ARM64. This alone halves the prediction time of medium networks.

**Cache-friendly batches.** Training a mini-batch processes the whole batch layer by layer: every weight row is loaded into the cache once and used for all samples, instead of streaming the full weight matrix once per sample. The backward pass walks the weights row by row as well, where it used to jump through memory column by column.

**Deterministic parallelism.** Large layers are split across goroutines: the forward pass and the weight gradients by output neuron, the backward pass by sample. Every number is computed by exactly one goroutine, in the same order as on one core, so the trained network is bit for bit identical for any number of workers (a test checks this). Layers with fewer than 32,768 multiply-adds per batch stay on one goroutine, because starting goroutines would cost more than it saves.

**Small networks stay small.** Networks whose layers have at most 256 weights train sample by sample, where every vector fits into a few CPU registers, and use inlined loops instead of function calls for rows shorter than 8 values.

**No garbage.** Training steps reuse their buffers and allocate nothing. A `Predictor` keeps its activation buffers between calls. `Network.Predict` takes them from a `sync.Pool`, so it stays safe for concurrent use.

## Using it well

```go
// Prediction on a hot path: one Predictor per goroutine, no allocations.
p := network.NewPredictor()
out, err := p.Predict(x)      // out is overwritten by the next call
label, err := p.Classify(x)

// Training on all cores (the default) or on one.
h, err := nn.Fit(ctx, network, train, val, nn.FitConfig{Workers: 0}) // 0 = GOMAXPROCS
```

- `Network.Predict` and `Network.Classify` are safe from many goroutines at once. Training a network, a `Predictor` and an `online.Online` are for one goroutine at a time.
- `nn.Network32` (float32) needs half the memory; on the M1 it is as fast as float64. See [Embedded systems](embedded.md).

## Robustness

Speed must not cost reliability. gopher-learn protects itself and the caller against bad data:

- **A bad batch cannot destroy a network.** If a loss or gradient becomes NaN or infinite, the step is skipped and `nn.ErrNonFinite` returned. `nn.Fit` then stops with `StopDiverged` and keeps the last finite weights; the online learner skips the sample.
- **Gradient clipping:** `FitConfig.ClipNorm` limits the length of every update, which keeps outliers from blowing up the weights.
- **Validated loading:** every reader in `persist` checks what it decodes: network shapes, data sets (sizes, finite values, consistent classes), online learners (configuration, memory), models, and the float32 binary format, which checks every size before allocating. Corrupt or malicious files fail with an error, not with a panic or an out-of-memory crash later. `persist.MaxDocumentSize` (default 1 GiB) caps the size of a document.
- **Fuzzing:** CI feeds random input to every reader, the CSV and LIBSVM loaders, the record reader, model prediction and the drift detector.

## Regularization and initialization

```go
net, err := nn.New(nn.Config{
	Inputs: 20, Hidden: []int{64, 64}, Outputs: 3,
	HiddenActivation: nn.ReLU, OutputActivation: nn.Softmax,
	Dropout:   0.2,  // share of hidden outputs dropped while training
	LayerNorm: true, // normalize every hidden layer per sample
	Init:      nn.HeNormal,
})
```

- **Dropout** is inverted: kept outputs are scaled up during training, so predictions need no correction and cost nothing extra.
- **Layer normalization** learns a gain and a shift per neuron; it is part of the float32 model too.
- **Init** chooses `GlorotUniform`, `GlorotNormal`, `HeUniform` or `HeNormal`. The default picks He for ReLU layers and Glorot otherwise.

`model.Config` passes `Dropout`, `LayerNorm` and `Init` through. The gradients of all three are checked against numerical gradients in the tests.

## Benchmarks in CI

Every pull request that changes Go code benchmarks the hot paths (prediction, `Predictor`, float32 `Predictor32`, training batches and epochs, `model.Predict`, `online.Learn` and the drift detectors) on the base branch and on the change and compares them with `internal/cmd/benchcheck`. The table appears in the job summary; the job fails if the geometric mean gets more than 10 % slower or a single benchmark more than 50 %. Run the same comparison locally:

```bash
go test -run='^$' -bench=. -count=6 ./nn/ > new.txt
```

```bash
go run ./internal/cmd/benchcheck -base old.txt -head new.txt
```
