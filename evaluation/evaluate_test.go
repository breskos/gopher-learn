package evaluation

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

// identity returns a network whose outputs equal its inputs (linear, identity weights).
func identity(t *testing.T, size int, labels []string) *nn.Network {
	t.Helper()
	n, err := nn.New(nn.Config{Inputs: size, Outputs: size, Labels: labels, OutputActivation: nn.Linear})
	if err != nil {
		t.Fatal(err)
	}
	l := n.Layers[0]
	clear(l.Weights)
	clear(l.Biases)
	for i := range size {
		l.Weights[i*size+i] = 1
	}
	return n
}

func TestEvaluateClassification(t *testing.T) {
	set := learn.NewSet(nn.Classification)
	_ = set.AddClass("a")
	_ = set.AddClass("b")
	_ = set.Add(learn.NewClassificationSample([]float64{1, 0}, "a"))     // predicted a
	_ = set.Add(learn.NewClassificationSample([]float64{0, 1}, "b"))     // predicted b
	_ = set.Add(learn.NewClassificationSample([]float64{0.9, 0.1}, "b")) // predicted a
	e, err := Evaluate(identity(t, 2, set.Labels), set, 0)
	if err != nil {
		t.Fatal(err)
	}
	if e.Correct != 2 || e.Wrong != 1 {
		t.Fatalf("correct/wrong = %d/%d", e.Correct, e.Wrong)
	}
	if e.Confusion["b"]["a"] != 1 {
		t.Fatalf("confusion = %v", e.Confusion)
	}
	// distance: only the wrong sample is off by 0.9 in both outputs → 0.5 × (0.81 + 0.81)
	if d := e.TotalDistance; d < 0.809 || d > 0.811 {
		t.Fatalf("distance = %v", d)
	}
	if e.OverallAccuracy() <= 0 || e.OverallBalancedAccuracy() <= 0 {
		t.Fatal("overall metrics must be positive")
	}
}

func TestEvaluateRegression(t *testing.T) {
	set := learn.NewSet(nn.Regression)
	_ = set.Add(learn.NewRegressionSample([]float64{0.5}, 0.52)) // error 0.02
	_ = set.Add(learn.NewRegressionSample([]float64{0.5}, 0.9))  // error 0.4
	e, err := Evaluate(identity(t, 1, nil), set, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	if e.Correct != 1 || e.Wrong != 1 {
		t.Fatalf("correct/wrong = %d/%d", e.Correct, e.Wrong)
	}
	var buf bytes.Buffer
	if err := e.WriteRegressionSummary(&buf); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "correct: 1  wrong: 1") {
		t.Fatalf("summary = %q", buf.String())
	}
}

func TestEvaluateDimensionError(t *testing.T) {
	set := learn.NewSet(nn.Regression)
	_ = set.Add(learn.NewRegressionSample([]float64{1, 2}, 0.5))
	if _, err := Evaluate(identity(t, 1, nil), set, 0.1); !errors.Is(err, nn.ErrDimension) {
		t.Fatalf("want ErrDimension, got %v", err)
	}
	if _, err := New(nn.Regression, nil).AddDistance([]float64{1}, []float64{1, 2}); !errors.Is(err, nn.ErrDimension) {
		t.Fatalf("AddDistance: want ErrDimension, got %v", err)
	}
}

func TestAddWithoutConstructor(t *testing.T) {
	var e Evaluation
	e.Add("x", "y")
	if e.Wrong != 1 || e.Confusion["x"]["y"] != 1 {
		t.Fatalf("zero-value evaluation: %+v", e)
	}
}

func TestAllMetricsInRange(t *testing.T) {
	e := sample()
	for _, c := range e.Classes() {
		for name, v := range map[string]float64{
			"specificity":  e.Specificity(c),
			"fallout":      e.Fallout(c),
			"fdr":          e.FalseDiscoveryRate(c),
			"npv":          e.NegativePredictiveValue(c),
			"informedness": (e.Informedness(c) + 1) / 2,
			"markedness":   (e.Markedness(c) + 1) / 2,
		} {
			if v < 0 || v > 1 {
				t.Errorf("%s(%s) = %v out of range", name, c, v)
			}
		}
	}
}

type failWriter struct{}

func (failWriter) Write([]byte) (int, error) { return 0, errors.New("disk full") }

func TestReportWriteErrors(t *testing.T) {
	e := sample()
	if err := e.WriteSummaries(failWriter{}); err == nil {
		t.Error("WriteSummaries: expected error")
	}
	if err := e.WriteConfusionMatrix(failWriter{}); err == nil {
		t.Error("WriteConfusionMatrix: expected error")
	}
}
