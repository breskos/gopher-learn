package evaluation

import (
	"errors"
	"math"

	"github.com/breskos/gopher-learn/nn"
)

// Prequential evaluates a model that learns from a stream with the
// test-then-train protocol: every sample is first predicted, the prediction
// is recorded here, and only then is the model trained on it. Each sample is
// thus evaluated before the model has seen it.
//
// Besides the cumulative Evaluation it keeps a sliding window of the most
// recent results, which shows how well the model does now, e.g. after a
// concept drift.
type Prequential struct {
	// Cumulative holds all results since the start.
	Cumulative *Evaluation `json:"cumulative"`
	// Window is the number of recent results the windowed metrics use.
	Window int `json:"window"`

	// Errors holds the recent per-sample errors in a ring buffer: 0 or 1
	// for classification, the absolute error for regression.
	Errors []float64 `json:"errors"`
	Next   int       `json:"next"`
	Sum    float64   `json:"sum"`
	SumSq  float64   `json:"sum_sq"`
}

// NewPrequential returns a prequential evaluation with a window of the
// given size. labels are the classes for classification.
func NewPrequential(task nn.Task, labels []string, window int) (*Prequential, error) {
	if window < 1 {
		return nil, errors.New("evaluation: window must be at least 1")
	}
	return &Prequential{Cumulative: New(task, labels), Window: window}, nil
}

func (p *Prequential) push(e float64) {
	if len(p.Errors) < p.Window {
		p.Errors = append(p.Errors, e)
	} else {
		old := p.Errors[p.Next]
		p.Sum -= old
		p.SumSq -= old * old
		p.Errors[p.Next] = e
		p.Next = (p.Next + 1) % p.Window
	}
	p.Sum += e
	p.SumSq += e * e
	// Recompute now and then to keep rounding errors from accumulating.
	if p.Next == 0 && len(p.Errors) == p.Window {
		p.Sum, p.SumSq = 0, 0
		for _, x := range p.Errors {
			p.Sum += x
			p.SumSq += x * x
		}
	}
}

// AddClassification records a classification prediction: the network output
// (scores, used for log loss and AUC; may be nil), the predicted and the
// actual class index. It returns the error of the sample (0 or 1), which can
// be fed to a drift detector.
func (p *Prequential) AddClassification(scores []float64, predicted, actual int) (float64, error) {
	labels := p.Cumulative.Labels
	if predicted < 0 || predicted >= len(labels) || actual < 0 || actual >= len(labels) {
		return 0, errors.New("evaluation: class index out of range")
	}
	if scores != nil {
		if err := p.Cumulative.AddScores(scores, actual); err != nil {
			return 0, err
		}
	}
	p.Cumulative.Add(labels[actual], labels[predicted])
	e := 0.0
	if predicted != actual {
		e = 1
	}
	p.push(e)
	return e, nil
}

// AddRegression records a regression prediction and returns its absolute
// error.
func (p *Prequential) AddRegression(actual, predicted float64) float64 {
	p.Cumulative.AddRegression(actual, predicted)
	e := math.Abs(predicted - actual)
	p.push(e)
	return e
}

// Len returns the number of results in the window.
func (p *Prequential) Len() int { return len(p.Errors) }

// Accuracy returns the share of correct classifications in the window.
func (p *Prequential) Accuracy() float64 {
	if len(p.Errors) == 0 {
		return 0
	}
	return 1 - p.Sum/float64(len(p.Errors))
}

// MAE returns the mean absolute regression error in the window.
func (p *Prequential) MAE() float64 {
	if len(p.Errors) == 0 {
		return 0
	}
	return p.Sum / float64(len(p.Errors))
}

// RMSE returns the root mean squared regression error in the window.
func (p *Prequential) RMSE() float64 {
	if len(p.Errors) == 0 {
		return 0
	}
	return math.Sqrt(max(p.SumSq, 0) / float64(len(p.Errors)))
}
