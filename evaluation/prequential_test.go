package evaluation

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/breskos/gopher-learn/nn"
)

func TestPrequentialClassification(t *testing.T) {
	p, err := NewPrequential(nn.Classification, []string{"a", "b"}, 4)
	if err != nil {
		t.Fatal(err)
	}
	if p.Accuracy() != 0 || p.MAE() != 0 || p.RMSE() != 0 {
		t.Fatal("empty window")
	}
	// predicted, actual
	for _, c := range [][2]int{{0, 0}, {1, 0}, {1, 1}, {1, 1}, {0, 0}, {0, 0}} {
		if _, err := p.AddClassification([]float64{0.6, 0.4}, c[0], c[1]); err != nil {
			t.Fatal(err)
		}
	}
	// window holds the last four: all correct
	if p.Len() != 4 || p.Accuracy() != 1 {
		t.Fatalf("window accuracy %.2f over %d", p.Accuracy(), p.Len())
	}
	if c := p.Cumulative.CorrectRatio(); math.Abs(c-5.0/6) > 1e-12 {
		t.Fatalf("cumulative %.3f", c)
	}
	if e, _ := p.AddClassification(nil, 1, 0); e != 1 || p.Accuracy() != 0.75 {
		t.Fatalf("error %v, accuracy %.2f", e, p.Accuracy())
	}
	if _, err := p.AddClassification(nil, 2, 0); err == nil {
		t.Error("out of range: expected error")
	}
	if _, err := p.AddClassification([]float64{1}, 1, 1); err == nil {
		t.Error("short scores: expected error")
	}
	if _, err := NewPrequential(nn.Classification, nil, 0); err == nil {
		t.Error("zero window: expected error")
	}
}

func TestPrequentialRegression(t *testing.T) {
	p, _ := NewPrequential(nn.Regression, nil, 2)
	p.AddRegression(1, 5) // error 4, drops out of the window
	p.AddRegression(1, 2)
	if e := p.AddRegression(0, -3); e != 3 {
		t.Fatalf("error %v", e)
	}
	if p.MAE() != 2 || math.Abs(p.RMSE()-math.Sqrt(5)) > 1e-12 {
		t.Fatalf("MAE %v, RMSE %v", p.MAE(), p.RMSE())
	}
	if p.Cumulative.Count != 3 {
		t.Fatal("cumulative count")
	}

	b, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	var q Prequential
	if err := json.Unmarshal(b, &q); err != nil {
		t.Fatal(err)
	}
	q.AddRegression(0, 1)
	p.AddRegression(0, 1)
	if q.MAE() != p.MAE() {
		t.Fatal("restored evaluation differs")
	}
}

func TestPrequentialLongRun(t *testing.T) {
	p, _ := NewPrequential(nn.Regression, nil, 7)
	for i := range 10000 {
		p.AddRegression(0, 0.1*float64(i%13))
	}
	var sum float64
	for _, e := range p.Errors {
		sum += e
	}
	if math.Abs(p.Sum-sum) > 1e-9 {
		t.Fatalf("running sum %.12f, actual %.12f", p.Sum, sum)
	}
}
