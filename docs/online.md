# Online learning

The `online` package trains a network on the fly. Samples arrive one by one and the network adapts, which suits streams, services that keep learning and autonomous agents. The [`drift`](../drift) package detects when the data changes.

## Learn from a stream

```go
// The set defines the task and the classes; it may start empty.
set := learn.NewSet(nn.Classification)
_ = set.AddClass("ok")
_ = set.AddClass("failing")

cfg := online.DefaultConfig()
cfg.HiddenActivation, cfg.OutputActivation = nn.Tanh, nn.Softmax
cfg.OnDrift = func(e online.DriftEvent) {
	log.Printf("drift after %d samples, forgot %d old ones", e.Sample, e.Forgotten)
}
o, err := online.New(set, 4, []int{12}, cfg)
if err != nil {
	return err
}

for sample := range samples { // e.g. from a message queue
	r, err := o.Learn(sample)
	if err != nil {
		return err
	}
	_ = r.Label // the prediction made before training on the sample
	if o.Learned%250 == 0 {
		if _, err := o.Iterate(ctx); err != nil && !errors.Is(err, online.ErrNotEnoughData) {
			return err
		}
	}
}
log.Printf("accuracy on the last %d samples: %.3f", o.Prequential.Len(), o.Prequential.Accuracy())
```

`Learn` follows the **test-then-train** protocol for every sample:

1. **Predict** the sample with the current network. The result contains the output, the predicted label and the error (0 or 1 for classification, the absolute error for regression).
2. **Evaluate:** the result goes into `o.Prequential`. Because each sample is evaluated before the network has seen it, this measures how well the model does on new data, not how well it remembers. `Prequential.Accuracy`, `MAE` and `RMSE` cover the last `Config.PrequentialWindow` samples, `Prequential.Cumulative` everything since the start.
3. **Detect drift:** the error goes into the drift detector. A significantly *rising* error is a drift; a falling error only means the model got better.
4. **Remember** the sample, within the bounds of the memory.
5. **Train** on the sample with `Config.HotShots` quick steps.

`Iterate` retrains on the remembered samples and returns the macro F1 score (classification) or correct ratio (regression) on a held-out split. Call it regularly; the hot shots alone adapt quickly but not thoroughly.

Prefer `nn.Tanh` with `nn.Softmax` outputs for online classification: sigmoid outputs trained on squared error saturate and can get stuck after a change of concept.

## Memory

| `Config.Memory` | Keeps | Suited for |
| --- | --- | --- |
| `online.Window` (default) | the `MemorySize` most recent samples | data that changes over time |
| `online.Reservoir` | a uniform random sample of `MemorySize` samples from the whole stream | stable data where old samples stay valuable |
| `online.Unbounded` | every sample | small streams, the behavior before v0.6 |

## Concept drift

| `Config.Drift` | Detector | Behavior |
| --- | --- | --- |
| `online.ADWIN` (default) | `drift.ADWIN` | adaptive window; detects abrupt and gradual changes and estimates how many recent samples belong to the new concept |
| `online.PageHinkley` | `drift.PageHinkley` | cumulative test for rising errors; cheap and fast on abrupt changes |
| `online.NoDrift` | none | |

`Config.DriftDelta` tunes the detector (ADWIN's confidence, default 0.002, or Page-Hinkley's threshold, default 50).

On a drift the learner **forgets the samples from before the change**: with ADWIN it keeps as many recent samples as ADWIN's window holds after the change, with Page-Hinkley `MinimumDataPoints`; a reservoir has no time order and is cleared. One change can make ADWIN cut its window several times while the new concept settles; the memory follows every cut, but `OnDrift` is called and `Result.Drift` is set only once per `Config.DriftCooldown` samples (default 200).

On the synthetic streams in `online/` ([gopher-learn-data](https://github.com/breskos/gopher-learn-data/tree/main/synthetic)), drift detection finds the change within about 40 samples and the prequential accuracy after an abrupt drift reaches 0.92 instead of 0.72 without it.

The detectors work on any stream of numbers and can be used on their own:

```go
d := drift.NewADWIN(0.002)
for _, latency := range latencies { // values in [0, 1] give the best guarantees
	if d.Add(latency) && d.Change > 0 {
		log.Printf("latency rose to %.3f", d.Mean())
	}
}
```

## Inject

`Inject` remembers a sample and trains on it without evaluating it first, e.g. to load known data. It rejects samples whose feature vector already exists unless `force` is set. If the set already contains data, `Init` trains until `Config.TargetScore` is reached.

## Persistence

The complete learner, including its memory, the prequential evaluation and the state of the drift detector, is saved and loaded with `persist.SaveOnline` and `persist.LoadOnline`. Learners saved before v0.6 load with unbounded memory and no drift detection.

## Config

`online.DefaultConfig()` fits small classification tasks. For regression, set `OutputActivation = nn.Linear` and `RegressionThreshold` to the error you accept. `Seed` makes runs reproducible.
