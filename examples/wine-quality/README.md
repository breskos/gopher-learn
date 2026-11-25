# Wine quality

Train a regressor that predicts the quality of red wine from physicochemical measurements ([Cortez et al., 2009](http://archive.ics.uci.edu/ml/datasets/Wine+Quality)).
The data with the target scaled to [0, 1] is available in [gopher-learn-data](https://github.com/breskos/gopher-learn-data/tree/master/wine-quality).

A prediction counts as correct if it is within `RegressionThreshold` of the target. Raise the threshold to make the evaluation more tolerant.

```
go run . -data path/to/winequality-red.csv
```

The input features are not scaled in this data set (some columns range up to several hundred), which limits what a sigmoid network can learn. Scaling the features to a similar range improves the results considerably.
