# Online learning

The `online` package trains a network on the fly. Data points are added one by one and the network adapts, which makes it suitable for streams and autonomous agents.

## Usage

```go
// The set defines the task and the classes; it may start empty.
set := learn.NewSet(nn.Classification)
_ = set.AddClass("R")
_ = set.AddClass("M")

o, err := online.New(set, 60, []int{100}, online.DefaultConfig())
if err != nil {
	return err
}

// If the set already contains data, Init trains until Config.TargetScore is reached.
score, err := o.Init(ctx)
```

New data points are added with `Inject`. It trains the network a few times on the new sample ("hot shots", see `Config.HotShots` and `Config.HotShotRate`) without retraining on the whole set. This trades learning quality for speed, so call `Iterate` regularly:

```go
sample := learn.NewClassificationSample([]float64{ /* 60 features */ }, "R")
if err := o.Inject(sample, false); err != nil && !errors.Is(err, online.ErrSampleExists) {
	return err
}

// Iterate retrains on a random split of the collected data and returns
// the macro F1 score (classification) or correct ratio (regression).
score, err := o.Iterate(ctx)
if errors.Is(err, online.ErrNotEnoughData) {
	// fewer than Config.MinimumDataPoints samples so far
}
```

`Inject` rejects samples whose feature vector already exists unless `force` is set.

The complete learner, including its data set, is saved and loaded with `persist.SaveOnline` and `persist.LoadOnline`.

## Config

`online.DefaultConfig()` fits small classification tasks. For regression, set `RegressionThreshold` to the error you accept. `Seed` makes runs reproducible.
