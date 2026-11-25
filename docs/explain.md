# Explainability

A model that predicts well is not automatically a model you can trust. The `explain` package answers the first question every user, auditor or colleague asks: **which inputs does the model rely on?** It does so with *permutation feature importance*, which works for every model gopher-learn trains, needs no retraining and gives results in the units of a metric you already know.

## In short

```go
imp, err := explain.Permutation(network, testSet, explain.Config{})
if err != nil {
	return err
}
for _, f := range imp {
	fmt.Printf("%-10s %.3f ± %.3f\n", f.Feature, f.Mean, f.Std)
}
```

For a model trained on raw records (see [Models](model.md)):

```go
imp, err := m.Importance(testRecords, explain.Config{})
```

## How it works

1. Evaluate the model on test data: this is the **baseline** score, e.g. 95 % accuracy.
2. Take one feature and **shuffle its values across the samples**. Every sample now carries the value of a random other sample for this feature. The distribution of the feature stays the same, but its relation to the target is gone.
3. Evaluate again. If the model relied on the feature, it gets worse; if it ignored it, nothing changes.
4. The **importance** is the loss: baseline minus shuffled score. Repeat the shuffle a few times (`Config.Repeats`, default 5) and report the mean and the standard deviation.

Every feature is shuffled on its own while all others keep their real values. The model is never retrained.

## Reading the result

The result is a list of `explain.Importance`, sorted from most to least important:

| Field | Meaning |
| --- | --- |
| `Feature` | name from `learn.Set.Features` (e.g. the CSV header) or the column name of a model; `x1`, `x2`, … without names |
| `Index` | position of the feature in the input |
| `Mean` | average loss of quality when the feature is shuffled, in units of the metric |
| `Std` | standard deviation of the loss over the repeats |
| `Drops` | the loss of every repeat |

**The unit is the metric.** With accuracy (the default for classification), a `Mean` of 0.15 means the model gets 15 percentage points worse without this feature. With RMSE (the default for regression), it means the root mean squared error grows by 0.15 in the units of the target. The sign is oriented so that **larger always means more important**, also for metrics where lower is better.

**Zero means unused.** Values around zero (within about two standard deviations) mean the model does not rely on the feature. Small negative values are random noise: by chance, the shuffled data happened to suit the model slightly better. They are not an error.

### Example: a shop model

A `model.Model` predicts whether a customer buys, from a numeric score `x`, the income and a color preference. A column of pure random numbers (`noise`) was added on purpose. On 600 test records the model reaches 99.5 % accuracy; `Importance` with 3 repeats gives:

| Column | Mean | Std |
| --- | ---: | ---: |
| `income` | 0.326 | 0.021 |
| `x` | 0.243 | 0.015 |
| `color` | 0.186 | 0.009 |
| `noise` | 0.002 | 0.003 |

Shuffling the income costs 33 accuracy points; the model leans on it most. The noise column is correctly worthless. The categorical `color` column is measured as one column, even though it becomes several encoded inputs inside the model.

### Example: Friedman #1

In this benchmark the target depends on x1 to x5 only: `10 sin(π x1 x2) + 20 (x3 − 0.5)² + 10 x4 + 5 x5`; x6 to x10 are irrelevant. A network trained on 4,200 samples (R² 0.94 on the test data) gives, as RMSE increase:

| Feature | Mean | | Feature | Mean |
| --- | ---: | --- | --- | ---: |
| x4 | 0.0795 | | x8 | 0.0000 |
| x2 | 0.0718 | | x10 | -0.0000 |
| x1 | 0.0678 | | x7 | -0.0000 |
| x5 | 0.0317 | | x6 | -0.0001 |
| x3 | 0.0304 | | x9 | -0.0002 |

All five relevant features come first, all five irrelevant ones are at zero. Both examples run as tests (`model.TestImportance`, `internal/synth.TestImportanceFriedman`), so the documentation stays true.

## Recipes

### Find data leaks

A **leak** is an input that is only known *after* the outcome, for example a voucher that a shop issues only after a purchase. A model happily uses it and looks excellent in testing, but in production the value is not available yet. Leaks show up as one feature that is far more important than everything else.

The shop model above, trained without income but with a `voucher` column that is set for 90 % of the buyers:

| Column | Mean | Std |
| --- | ---: | ---: |
| `voucher` | 0.365 | 0.016 |
| `x` | 0.032 | 0.006 |
| `color` | 0.028 | 0.007 |
| `noise` | 0.001 | 0.001 |

Test accuracy: 95.3 %. Without the voucher column the honest accuracy is 76.8 %, and `x` (0.154) and `color` (0.112) carry the model. If one column dominates like this, check when its value becomes known. IDs, timestamps and columns derived from the target are the usual suspects. (`model.TestImportanceFindsLeak` checks this example.)

### Remove useless features

Features with an importance around zero can usually be dropped: the model gets smaller and faster, which matters on [embedded devices](embedded.md), and less prone to overfitting. Drop them, **retrain** and compare the test score, because the model may redistribute what it learned.

### Compare model versions

Compute the importance for the old and the new model on the same test data. A feature that suddenly matters much more (or no longer at all) explains changed predictions, and a large shift is worth a look before deploying.

### Check what an online learner uses

For a stream, compute the importance of the network of an `online.Online` on recent samples, for example the samples in `o.Data` after a detected drift: `explain.Permutation(o.Network, o.Data, cfg)`. If the important features changed with the drift, the world changed, not only the noise.

## Choosing the metric

`Config.Metric` accepts any `evaluation.Metric`. The default is accuracy for classification and RMSE for regression.

- For imbalanced classes use `evaluation.MacroF1` or `evaluation.BalancedAccuracy`: with 95 % of one class, a feature can matter a lot for the rare class without moving the accuracy.
- `evaluation.LogLoss` also sees changes in confidence that do not flip a prediction, so it shows smaller effects.
- Use the metric you care about in production; the importance answers "how much of *this* do I lose without the feature".

## Cost

Every importance computation evaluates the model `1 + features × repeats` times on the test data. Measured on an Apple M1 with a network of 32 hidden neurons, 1,000 test samples and 5 repeats:

| Features | Evaluations | 1 core | 8 cores |
| ---: | ---: | ---: | ---: |
| 10 | 51 | 35 ms | 16 ms |
| 50 | 251 | 198 ms | 61 ms |
| 200 | 1,001 | 2.0 s | 0.44 s |

The shop model (5 columns, 600 records, 5 repeats) takes 16 ms on one core and 9 ms on eight.

Features are evaluated in parallel on all cores; `Config.Workers` limits that, and the result does not depend on it (every feature has its own shuffle seed). To make it faster:

- Use fewer test samples: 1,000 to 5,000 are usually enough for stable values; check that `Std` is small compared with `Mean`.
- Use fewer repeats (`Repeats: 2` or `3`) for a first look, more for a report.
- Compute it once per model version, not per request.

## Limits

- **It explains the model, not the world.** A high importance means the model relies on the feature, not that the feature causes the outcome. A model may use a proxy, e.g. the postal code instead of the income.
- **It explains the model as a whole, not a single prediction.** It does not tell why one customer got a particular answer.
- **Correlated features share their importance.** If two features carry the same information, the model can fall back on one when the other is shuffled, so both look less important than they are. Look at such features together, or drop one and retrain to see the real effect.
- **Shuffling can create unrealistic samples**, e.g. a 30 m² flat with 9 rooms, when features depend on each other. The model is then judged partly on inputs it never sees in practice. With strongly dependent features, interpret the values with care.
- **Use test data.** On training data the importance also rewards what the model merely memorized, so overfitted features look more important than they are.

## Other models

`explain.Permute` computes the importance for anything you can evaluate: give it the feature names, the number of samples and a function that returns the score with one feature shuffled (row `i` takes the value of row `order[i]`; feature `-1` means unshuffled). `explain.PermuteParallel` does the same on several cores, with one score function per worker:

```go
imp, err := explain.Permute(names, rows, func(feature int, order []int) (float64, error) {
	return myScore(feature, order)
}, evaluation.Accuracy, explain.Config{})
```
