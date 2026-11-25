# Evaluation

`evaluation.Evaluate(network, set, threshold)` runs a network on a data set and collects the results. The engine and the online learner use it internally; `Winner()` and `LastEvaluation` expose the result.

## Classification metrics

All per-class metrics take the class label. The `Overall*` variants average over all classes (macro average).

| Method | Definition |
| --- | --- |
| `TruePositives`, `FalsePositives`, `TrueNegatives`, `FalseNegatives` | counts from the confusion matrix |
| `Accuracy` | (TP + TN) / (P + N) |
| `Recall` | TP / P (sensitivity) |
| `Specificity` | TN / N |
| `Precision` | TP / (TP + FP) |
| `Fallout` | FP / N (false positive rate) |
| `FalseDiscoveryRate` | FP / (FP + TP) |
| `NegativePredictiveValue` | TN / (TN + FN) |
| `FMeasure` | 2TP / (2TP + FP + FN) |
| `BalancedAccuracy` | (recall + specificity) / 2 |
| `Informedness` | recall + specificity − 1 |
| `Markedness` | precision + negative predictive value − 1 |

Metrics with a zero denominator return 0.

## Regression

A regression prediction counts as correct if its absolute error is below `Threshold`. `CorrectRatio` returns the share of correct predictions.

`TotalDistance` is the summed half squared error between outputs and targets for both tasks.

## Reports

`WriteConfusionMatrix`, `WriteSummary`, `WriteSummaries` and `WriteRegressionSummary` write human-readable reports to any `io.Writer`.
