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
for _, f := range imp {
	fmt.Printf("%-10s %.3f ± %.3f\n", f.Feature, f.Mean, f.Std)
}
```

The importance of a feature is how much worse the model gets when the values of that feature are shuffled across the samples. Shuffling keeps the distribution of the feature but breaks its relation to the target. A feature the model ignores costs nothing; a feature it depends on costs a lot.

- The result is sorted from most to least important. `Mean` is the average loss of quality in units of the metric (e.g. accuracy points), oriented so that larger means more important for every metric. Values around zero mean the model does not use the feature; small negative values are noise.
- `Config.Repeats` (default 5) shuffles every feature several times; `Std` shows how stable the value is.
- **Use data the model was not trained on.** On training data, importance also rewards features the model merely memorized.
- Names come from `learn.Set.Features` (e.g. a CSV header), otherwise `x1`, `x2`, ….

On Friedman #1, where only x1 to x5 influence the target, a trained network gives x1 to x5 importances between 0.03 and 0.08 (RMSE) and x6 to x10 importances of at most 0.0001.

### For models on raw records

`model.Model.Importance(records, cfg)` shuffles the raw values of a column across the records. Numeric and categorical columns are treated alike, so the importance of a categorical column covers all of its encoded features:

```go
imp, err := m.Importance(testRecords, explain.Config{})
```

### Correlated features

If two features carry the same information, the model can fall back on one when the other is shuffled, so both look less important than they are. Look at correlated features together, or remove one and retrain to see the effect.

### Other models

`explain.Permute` computes the importance from any score function. Use it for models that are not a network on a `learn.Set`:

```go
imp, err := explain.Permute(names, rows, func(feature int, order []int) (float64, error) {
	// evaluate the model with column `feature` taken from row order[i] for row i,
	// or unchanged for feature -1
}, evaluation.Accuracy, explain.Config{})
```
