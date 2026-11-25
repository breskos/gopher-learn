# Wine quality

Train a regressor that predicts the quality of red wine from physicochemical measurements ([Cortez et al., 2009](http://archive.ics.uci.edu/ml/datasets/Wine+Quality)).
The data with the target scaled to [0, 1] is available in [gopher-learn-data](https://github.com/breskos/gopher-learn-data/tree/master/wine-quality).

The example shows how to:

- hold back a test set the training never sees,
- standardize the features with a `learn.Scaler` fitted on the training data only,
- train with the engine and evaluate the winner on the test set.

A prediction counts as correct if it is within 0.04 of the target.

```
go run . -data path/to/winequality-red.csv
```

The raw features range from below 1 to several hundred. Standardizing them improves the test set results; compare with `-scale=false`:

| | correct within 0.04 | distance |
| --- | --- | --- |
| without scaling | 47.8 % | 0.645 |
| standardized | 54.1 % | 0.579 |

The quality is rated in steps of 0.1, so hitting it within 0.04 is demanding.
