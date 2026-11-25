# Engine

The engine trains networks on a data set and keeps the best one.
Instead of tuning learning rates and splitting data by hand, you hand the engine a `learn.Set` and a `Config`.

```go
cfg := engine.DefaultConfig()
cfg.Seed = 42 // reproducible runs
cfg.Logger = slog.New(slog.NewTextHandler(os.Stderr, nil))

e, err := engine.New(data, []int{100}, cfg)
if err != nil {
	return err
}
if err := e.Run(ctx, engine.FMeasure); err != nil {
	return err
}
network, eval := e.Winner()
```

`Run` can be cancelled through the context; the best network found so far stays available through `Winner`.

## Training loop

```
split data into training and validation (Config.TrainingSplit)
for each try (Config.Tries), starting from a freshly initialized network:
    for learningRate := LearningRate; learningRate > 0; learningRate -= Decay:
        train Epochs epochs on the shuffled training set (mini-batches of BatchSize)
        evaluate on the validation set
        keep a copy of the network if it beats the best one so far
```

## Criteria

| Criterion | Picks the network with the highest … | Suited for |
| --- | --- | --- |
| `Accuracy` | mean per-class accuracy | balanced classification |
| `BalancedAccuracy` | mean balanced accuracy | unbalanced classification |
| `FMeasure` | macro F1 score | unbalanced classification |
| `Simple` | ratio of correct predictions | regression with threshold |
| `Distance` | lowest distance to the targets | regression, classification |

## Config

The defaults fit most small classification tasks. For regression, set `RegressionThreshold` and tune `LearningRate` and `Decay` to your data.
`HiddenActivation` and `OutputActivation` select the activation functions (`nn.Sigmoid`, `nn.Tanh`, `nn.ReLU`, `nn.Linear`, `nn.Softmax`); both default to sigmoid.
For classification, `nn.Softmax` on the output layer trains with cross-entropy and usually converges faster.
