# Model selection and explainability

Two questions come up once a model works: *Which settings work best?* and *Which inputs does the model rely on?* The `tune` package answers the first with a cross-validated hyperparameter search, the `explain` package the second with permutation feature importance.

## Hyperparameter search

```go
space := tune.Space{
	Hidden:       [][]int{{16}, {32}, {32, 16}},
	LearningRate: []float64{0.003, 0.01, 0.03},
	Dropout:      []float64{0, 0.2},
}
res, err := tune.Search(ctx, data, space.Grid(), tune.Config{Metric: evaluation.MacroF1})
if err != nil {
	return err
}
for _, t := range res.Ranked() {
	fmt.Printf("%-50v %.3f ± %.3f\n", t.Candidate, t.Mean, t.Std)
}

// train the winner on all data
network, scaler, err := tune.Fit(ctx, data, res.Best().Candidate, tune.Config{})
```

### The space

A `tune.Space` lists the values to try for each hyperparameter: hidden layer sizes, hidden activation, learning rate, batch size, dropout, weight decay and layer normalization. Fields left empty use one default (one hidden layer of 32 tanh neurons, learning rate 0.01, batch size 32, no regularization).

- `space.Grid()` returns every combination. Three values each for three parameters are 27 candidates.
- `space.Random(n, rng)` returns `n` different combinations drawn at random. For more than two or three parameters, random search usually finds an equally good configuration with a fraction of the trials, because only a few parameters matter much.

You can also build the `[]tune.Candidate` list yourself.

### How a candidate is evaluated

1. The data is split into `Config.Folds` folds (default 5), stratified by class for classification. **All candidates use the same folds**, so their scores are directly comparable.
2. For every fold, the features are standardized with a scaler fitted **on the training part only**, so no information leaks from the test part into the training.
3. A network is trained with `nn.Fit` (Adam by default) on the training part. `Config.ValidationRatio` (default 10 %) of it is held out for early stopping after `Config.Patience` epochs without improvement (default 10; negative disables it).
4. The network is evaluated on the test fold with `Config.Metric`.

A `tune.Trial` holds the score of every fold, their mean and standard deviation and the number of epochs trained. A small standard deviation means the result is stable; if two candidates differ by less than their standard deviations, prefer the simpler one.

### Metrics

`Config.Metric` takes any `evaluation.Metric`. Predefined:

| Task | Metric | Better |
| --- | --- | --- |
| classification | `evaluation.Accuracy` (default), `MacroF1`, `BalancedAccuracy`, `MacroAUC` | higher |
| classification | `evaluation.LogLoss` | lower |
| regression | `evaluation.RMSE` (default), `MAE` | lower |
| regression | `evaluation.R2` | higher |

For imbalanced classes prefer `MacroF1` or `BalancedAccuracy`; accuracy rewards always predicting the majority class.

### Speed and reproducibility

Trials run in parallel on all cores (`Config.Workers`); each training then uses one core. Every trial and fold has its own seed derived from `Config.Seed`, so the result is identical for any number of workers and repeatable. `Config.OnTrial` reports finished trials, and the search can be cancelled through the context.

On the synthetic spirals (1,500 samples, 3 folds, 6 random candidates) the search takes about one second on an Apple M1 and ranks the candidates from 99.8 % (32 neurons, learning rate 0.03) down to 40 % (4 neurons, learning rate 0.001).

## Permutation feature importance

```go
imp, err := explain.Permutation(network, testSet, explain.Config{})
```

The importance of a feature is how much worse the model gets when the values of that feature are shuffled across the samples. See [Explainability](explain.md) for how to read the results, recipes (finding data leaks, removing useless features, comparing model versions), cost and limits.
