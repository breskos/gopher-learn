package evaluation

import (
	"fmt"
	"slices"
)

// Metric scores an evaluation with a single number, e.g. to compare models
// in a hyperparameter search or to measure feature importance.
type Metric struct {
	// Name identifies the metric, e.g. "accuracy".
	Name string
	// Higher reports whether larger values are better.
	Higher bool
	// Value computes the metric.
	Value func(*Evaluation) float64
}

// Better reports whether score a is better than score b.
func (m Metric) Better(a, b float64) bool {
	if m.Higher {
		return a > b
	}
	return a < b
}

// The predefined metrics. Classification: Accuracy, MacroF1,
// BalancedAccuracy, MacroAUC, LogLoss. Regression: RMSE, MAE, R2.
var (
	Accuracy         = Metric{"accuracy", true, (*Evaluation).CorrectRatio}
	MacroF1          = Metric{"macro-f1", true, (*Evaluation).OverallFMeasure}
	BalancedAccuracy = Metric{"balanced-accuracy", true, (*Evaluation).OverallBalancedAccuracy}
	MacroAUC         = Metric{"macro-auc", true, (*Evaluation).MacroAUC}
	LogLoss          = Metric{"log-loss", false, (*Evaluation).LogLoss}
	RMSE             = Metric{"rmse", false, (*Evaluation).RMSE}
	MAE              = Metric{"mae", false, (*Evaluation).MAE}
	R2               = Metric{"r2", true, (*Evaluation).R2}
)

var metrics = []Metric{Accuracy, MacroF1, BalancedAccuracy, MacroAUC, LogLoss, RMSE, MAE, R2}

// MetricByName returns the predefined metric with the given name.
func MetricByName(name string) (Metric, error) {
	i := slices.IndexFunc(metrics, func(m Metric) bool { return m.Name == name })
	if i < 0 {
		return Metric{}, fmt.Errorf("evaluation: unknown metric %q", name)
	}
	return metrics[i], nil
}
