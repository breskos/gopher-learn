// Package evaluation measures the quality of classification and regression
// networks: confusion matrix, per-class metrics and distance to the target.
package evaluation

import (
	"fmt"
	"math"
	"slices"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

// Evaluation collects predictions and derives metrics from them.
type Evaluation struct {
	Task nn.Task `json:"task"`
	// Confusion counts predictions: Confusion[actual][predicted].
	Confusion map[string]map[string]int `json:"confusion,omitempty"`
	Correct   int                       `json:"correct"`
	Wrong     int                       `json:"wrong"`
	// TotalDistance is the summed half squared error between outputs and targets.
	TotalDistance float64 `json:"total_distance"`
	// Threshold is the maximum absolute error a regression prediction may
	// have to be counted as correct.
	Threshold float64 `json:"threshold"`

	// Labels holds the class labels in the order of the network outputs.
	Labels []string `json:"labels,omitempty"`
	// Regression sums for MAE, RMSE and R².
	Count       int     `json:"count,omitempty"`
	SumAbsError float64 `json:"sum_abs_error,omitempty"`
	SumSqError  float64 `json:"sum_sq_error,omitempty"`
	SumActual   float64 `json:"sum_actual,omitempty"`
	SumActualSq float64 `json:"sum_actual_sq,omitempty"`

	// scores and classes hold the network outputs of classification
	// predictions for log loss and ROC-AUC.
	scores  [][]float64
	classes []int
}

// New creates an empty evaluation for the given classes.
func New(task nn.Task, classes []string) *Evaluation {
	e := &Evaluation{Task: task, Labels: slices.Clone(classes), Confusion: make(map[string]map[string]int, len(classes))}
	for _, c := range classes {
		e.Confusion[c] = make(map[string]int, len(classes))
	}
	return e
}

// Evaluate runs the network on every sample of the set.
func Evaluate(n *nn.Network, set *learn.Set, threshold float64) (*Evaluation, error) {
	e := New(set.Task, set.Labels)
	e.Threshold = threshold
	for i, s := range set.Samples {
		out, err := n.Predict(s.Vector)
		if err != nil {
			return nil, fmt.Errorf("evaluation: sample %d: %w", i, err)
		}
		if _, err := e.AddDistance(out, s.Output); err != nil {
			return nil, fmt.Errorf("evaluation: sample %d: %w", i, err)
		}
		if set.Task == nn.Classification {
			e.Add(s.Label, set.Labels[nn.ArgMax(out)])
			if err := e.AddScores(out, s.Class); err != nil {
				return nil, fmt.Errorf("evaluation: sample %d: %w", i, err)
			}
		} else {
			e.AddRegression(s.Value, out[0])
		}
	}
	return e, nil
}

// AddScores records the network output of a classification prediction
// together with the index of the actual class. The scores are used for
// LogLoss and AUC; they are not persisted.
func (e *Evaluation) AddScores(scores []float64, actual int) error {
	if actual < 0 || actual >= len(scores) {
		return fmt.Errorf("evaluation: class index %d out of range for %d outputs", actual, len(scores))
	}
	e.scores = append(e.scores, slices.Clone(scores))
	e.classes = append(e.classes, actual)
	return nil
}

// Add records a classification prediction.
func (e *Evaluation) Add(actual, predicted string) {
	if e.Confusion == nil {
		e.Confusion = make(map[string]map[string]int)
	}
	if e.Confusion[actual] == nil {
		e.Confusion[actual] = make(map[string]int)
	}
	e.Confusion[actual][predicted]++
	if actual == predicted {
		e.Correct++
	} else {
		e.Wrong++
	}
}

// AddRegression records a regression prediction.
func (e *Evaluation) AddRegression(actual, predicted float64) {
	diff := predicted - actual
	e.Count++
	e.SumAbsError += math.Abs(diff)
	e.SumSqError += diff * diff
	e.SumActual += actual
	e.SumActualSq += actual * actual
	if math.Abs(diff) < e.Threshold {
		e.Correct++
	} else {
		e.Wrong++
	}
}

// AddDistance adds the half squared error between out and ideal to
// TotalDistance and returns it.
func (e *Evaluation) AddDistance(out, ideal []float64) (float64, error) {
	if len(out) != len(ideal) {
		return 0, fmt.Errorf("%w: output %d, target %d", nn.ErrDimension, len(out), len(ideal))
	}
	var d float64
	for i := range out {
		diff := out[i] - ideal[i]
		d += diff * diff
	}
	d /= 2
	e.TotalDistance += d
	return d, nil
}

// Classes returns all class labels seen as actual class, sorted.
func (e *Evaluation) Classes() []string {
	classes := make([]string, 0, len(e.Confusion))
	for c := range e.Confusion {
		classes = append(classes, c)
	}
	slices.Sort(classes)
	return classes
}

// CorrectRatio returns correct / (correct + wrong), or 0 without predictions.
func (e *Evaluation) CorrectRatio() float64 {
	return ratio(e.Correct, e.Correct+e.Wrong)
}

func ratio(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b)
}
