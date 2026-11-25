# Embedded systems

gopher-learn runs on small devices: single-board computers, industrial gateways, routers and anything else Go compiles for. The usual setup is to **train on a computer** and **predict on the device** with a float32 model that is small, loads in microseconds and allocates no memory while predicting.

## At a glance

| | |
| --- | --- |
| Dependencies | none: pure Go, standard library only, no cgo |
| Targets checked in CI | `linux/arm` (ARMv7), `linux/riscv64`, `js/wasm`, `wasip1/wasm`; tests run on 32-bit `386` |
| Also builds for | everything Go supports, e.g. `linux/arm64`, `linux/mips` |
| Program size | 1.5 to 1.9 MB for the example device program (stripped) |
| Model size | 4 bytes per parameter plus a few bytes per layer; 1 KB for a typical sensor model |
| Memory while predicting | the model plus one activation buffer per layer; no allocations per prediction |
| Model loading | 1.4 µs for an 867-parameter model, 8.7 µs for 6,302 parameters (Apple M1) |

TinyGo is not supported yet; it is planned for v0.12.

## The workflow

**1. Train on a computer** with the full library: `nn.Fit`, `model.Train`, the engine or the online learner.

**2. Convert to float32 and export:**

```go
small := network.Float32()           // inference-only copy, half the memory
data, err := small.MarshalBinary()   // compact binary format
err = os.WriteFile("motor.bin", data, 0o644)
```

**3. Predict on the device** with only the `nn` package:

```go
data, err := os.ReadFile("motor.bin")
var network nn.Network32
if err := network.UnmarshalBinary(data); err != nil {
	return err // corrupt or truncated files fail here, never later
}
p := network.NewPredictor() // create once, reuse for every reading
for reading := range sensor {
	label, err := p.Classify(reading) // no allocation
}
```

[examples/embedded](../examples/embedded) contains both programs: `train` learns a motor fault detector from simulated temperature, vibration, current and speed readings, `device` reads readings from standard input and prints `ok` or `fault`.

```bash
go run ./examples/embedded/train -out motor.bin
```

```bash
GOOS=linux GOARCH=arm GOARM=7 go build -trimpath -ldflags="-s -w" -o device ./examples/embedded/device
```

## Measured with the example

Fault detector with 4 inputs, hidden layers of 16 and 8 ReLU neurons and 2 outputs (234 parameters):

| | |
| --- | --- |
| Test accuracy, float64 / float32 | 99.5 % / 99.5 %, the same prediction for 100 % of 4,000 test readings |
| Model file | 1,011 bytes |
| Device program, stripped | 1.7 MB `linux/arm`, 1.6 MB `linux/arm64`, 1.5 MB `linux/riscv64`, 1.9 MB `linux/mips` |
| 1,000,000 readings, parsing included | 0.64 s on one core of an Apple M1 (1.5 million per second) |
| Peak memory of the process | 9.7 MB, almost all of it the Go runtime |

These numbers come from an Apple M1, which is itself an ARM64 CPU. A small ARM board is slower; it has not been measured yet, but even a hundred times slower would still classify more than 10,000 readings per second.

## Model sizes

| Network | Parameters | float32 model file | float64 JSON file | Loading float32 / JSON |
| --- | ---: | ---: | ---: | ---: |
| 4 × 8 × 2 | 58 | 276 B | 1.2 KB | 0.27 µs / 18 µs |
| 8 × 32 × 16 × 3 | 867 | 3.5 KB | 17 KB | 1.4 µs / 169 µs |
| 60 × 100 × 2 | 6,302 | 25 KB | 126 KB | 8.7 µs / 1.2 ms |
| 784 × 128 × 64 × 10 | 109,386 | 438 KB | 2.3 MB | 141 µs / 22 ms |

The float32 network keeps the same numbers in memory as in the file. Its outputs agree with the float64 network to about 1e-5, which practically never changes a classification.

## The binary format

`MarshalBinary` writes little-endian values: the magic `GLN1`, the number of layers and labels, then for every layer its inputs, outputs, activation and flags (layer normalization), its weights, biases and, if normalized, gains and shifts, and finally the labels. `UnmarshalBinary`

- checks every size against the remaining data **before** allocating, so a corrupt or malicious file cannot exhaust the memory,
- limits networks to 1,024 layers and 2^28 values and works the same on 32-bit and 64-bit platforms,
- rejects truncated files, trailing data and invalid activations, and validates the network,
- uses no reflection and no JSON parser.

It is fuzz-tested in CI.

## Settings for small systems

| Setting | Effect |
| --- | --- |
| `FitConfig.Workers = 1` | trains on the calling goroutine only, when training on the device |
| `online.Config.Memory = online.Window`, small `MemorySize` | bounds the memory of a learner that runs for months |
| `online.Config.Drift = online.PageHinkley` | the cheapest drift detector (9 ns per sample, constant memory) |
| `persist.MaxDocumentSize` | largest JSON document the loaders accept |
| `GOMEMLIMIT`, `GOGC` | memory limits of the Go runtime itself |

## Learning on the device

The online learner works on devices too: `online.Online.Learn` takes 2.4 µs per sample on the M1, its memory is bounded by `MemorySize`, and the drift detectors need constant (Page-Hinkley) or logarithmic (ADWIN) memory. Save its state with `persist.SaveOnline`: it writes to a temporary file, flushes it to the disk and renames it, so a power loss leaves either the old or the new state, never a broken file.
