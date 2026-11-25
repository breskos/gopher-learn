# Performance and embedded use

gopher-learn is pure Go without cgo and runs anywhere Go runs, from servers to ARM boards and WebAssembly. This page shows how fast it is, how to keep memory low and how it protects itself against bad data.

## Numbers

Apple M1, Go 1.27, median of six runs (`make bench`). Each epoch trains 1,024 samples with Adam on mini-batches of 32.

| Network | Predict | Predictor (no allocation) | float32 Predictor | Training epoch, 8 cores | Training epoch, 1 core |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny sensor model 4 × 8 × 2 | 134 ns | 123 ns | 134 ns | 0.22 ms | 0.22 ms |
| sonar 60 × 100 × 2 | 2.6 µs | 2.6 µs | 2.5 µs | 5.0 ms | 5.4 ms |
| MNIST-sized 784 × 128 × 64 × 10 | 31 µs | 32 µs | 32 µs | 37 ms | 69 ms |

Compared with v0.6, prediction of medium and large networks is 2 to 4 times faster and a training epoch of the MNIST-sized network 5 times faster (1 core: 2.8 times). Tiny networks are as fast as before. Other hot paths: `model.Model.Predict` from a raw record 0.7 µs, `online.Online.Learn` 2.4 µs, `drift.ADWIN.Add` 0.28 µs without allocation, `drift.PageHinkley.Add` 9 ns.

### What makes it fast

- **Dense kernels** with eight independent accumulators let the CPU overlap multiply-adds; Go turns them into fused multiply-add instructions on ARM64.
- **Batched training:** every weight row stays in the cache while all samples of a mini-batch pass it, and the backward pass walks the weights row by row.
- **Parallel layers:** large layers are split across goroutines by output neurons (forward pass, weight gradients) and by samples (backward pass). Every value is computed by exactly one goroutine in the same order, so the result is bit for bit the same for any number of workers. Small layers stay on one goroutine, where starting goroutines would cost more than it saves.
- **Small networks** train sample by sample, where every vector fits into registers.
- **No allocations** in training steps and in `Predictor`.

## Inference without allocations

`Network.Predict` is safe for concurrent use and allocates only the returned vector. On hot paths, give every goroutine its own `Predictor`:

```go
p := network.NewPredictor()
for x := range inputs {
	out, err := p.Predict(x) // out is reused by the next call
	label, err := p.Classify(x)
}
```

## float32 models for small devices

A trained network converts to an inference-only `nn.Network32` with half the memory. Its binary encoding loads without reflection or a JSON parser and checks every size before allocating:

```go
small := network.Float32()
data, err := small.MarshalBinary() // e.g. write to flash
// on the device
var n nn.Network32
if err := n.UnmarshalBinary(data); err != nil {
	return err
}
p := n.NewPredictor()
out, err := p.Predict([]float32{0.2, 0.7, 0.1, 0.9})
```

Outputs agree with the float64 network to about 1e-5. A 60 × 100 × 2 network needs 25 KB instead of 50 KB.

## Settings for small systems

| Setting | Effect |
| --- | --- |
| `nn.FitConfig.Workers = 1` (or `nn.Step.Workers`) | trains on the calling goroutine only |
| `online.Config.Memory = online.Window` with a small `MemorySize` | bounds the memory of a learner that runs forever |
| `persist.MaxDocumentSize` | largest file the loaders accept (default 1 GiB) |
| `GOMEMLIMIT`, `GOGC` | the Go runtime's own memory limits |

The packages build for `linux/arm`, `linux/riscv64`, `js/wasm` and `wasip1/wasm`, and the tests run on 32-bit `386` in CI.

## Robustness

- **No destroyed networks:** if a loss or gradient is NaN or infinite, the training step is skipped and `nn.ErrNonFinite` returned; `nn.Fit` stops with `StopDiverged` and keeps the last finite weights. The online learner skips such samples.
- **Gradient clipping:** `FitConfig.ClipNorm` limits the length of every update.
- **Validated loading:** every reader in `persist` checks what it decodes: network shapes, data sets (sizes, finite values, consistent classes), online learners (configuration, memory), models and the float32 binary format. Corrupt or malicious files fail with an error instead of a panic later on. Fuzz tests feed random documents to every reader in CI.
- **Concurrency:** prediction is safe from many goroutines; training a network, `Predictor` and `online.Online` are for one goroutine at a time.

## Regularization and initialization

```go
net, err := nn.New(nn.Config{
	Inputs: 20, Hidden: []int{64, 64}, Outputs: 3,
	HiddenActivation: nn.ReLU, OutputActivation: nn.Softmax,
	Dropout:   0.2,       // share of hidden outputs dropped while training
	LayerNorm: true,      // normalize every hidden layer per sample
	Init:      nn.HeNormal,
})
```

Dropout is inverted (kept outputs are scaled up while training), so predictions need no correction. Layer normalization learns a gain and a shift per neuron and is part of the float32 model too. `Init` chooses between `GlorotUniform`, `GlorotNormal`, `HeUniform` and `HeNormal`; the default picks He for ReLU layers and Glorot otherwise. `model.Config` passes `Dropout`, `LayerNorm` and `Init` through.

## Benchmarks in CI

Every pull request runs the benchmarks of `nn`, `drift`, `model` and `online` on the base branch and on the change and compares them with `internal/cmd/benchcheck`. The job fails if the geometric mean gets more than 10 % slower or a single benchmark more than 50 %; the table appears in the job summary.
