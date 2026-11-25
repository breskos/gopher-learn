package evaluation

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

func near(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("%s = %v, want %v", name, got, want)
	}
}

func TestRegressionMetrics(t *testing.T) {
	e := New(nn.Regression, nil)
	// actual 1, 2, 3; predicted 1.5, 2, 2
	for _, p := range [][2]float64{{1, 1.5}, {2, 2}, {3, 2}} {
		e.AddRegression(p[0], p[1])
	}
	near(t, "MAE", e.MAE(), 0.5)
	near(t, "RMSE", e.RMSE(), math.Sqrt(1.25/3))
	near(t, "R2", e.R2(), 1-1.25/2) // total sum of squares around the mean 2 is 2
	empty := New(nn.Regression, nil)
	if empty.MAE() != 0 || empty.RMSE() != 0 || empty.R2() != 0 {
		t.Error("empty evaluation must return 0")
	}
	constant := New(nn.Regression, nil)
	constant.AddRegression(1, 2)
	constant.AddRegression(1, 0)
	if constant.R2() != 0 {
		t.Errorf("R2 of constant targets = %v", constant.R2())
	}
}

func TestLogLossAndAUC(t *testing.T) {
	e := New(nn.Classification, []string{"neg", "pos"})
	// scores for class "pos": 0.9, 0.8 (pos), 0.7 (neg), 0.6 (pos), 0.2 (neg)
	// pairs pos>neg: (0.9>0.7,0.9>0.2,0.8>0.7,0.8>0.2,0.6>0.2) = 5 of 6
	for _, s := range []struct {
		p     float64
		class int
	}{{0.9, 1}, {0.8, 1}, {0.7, 0}, {0.6, 1}, {0.2, 0}} {
		if err := e.AddScores([]float64{1 - s.p, s.p}, s.class); err != nil {
			t.Fatal(err)
		}
	}
	near(t, "AUC(pos)", e.AUC("pos"), 5.0/6)
	near(t, "AUC(neg)", e.AUC("neg"), 5.0/6)
	near(t, "MacroAUC", e.MacroAUC(), 5.0/6)
	want := -(math.Log(0.9) + math.Log(0.8) + math.Log(0.3) + math.Log(0.6) + math.Log(0.8)) / 5
	near(t, "LogLoss", e.LogLoss(), want)

	if e.AUC("unknown") != 0 {
		t.Error("AUC of unknown class must be 0")
	}
	if err := e.AddScores([]float64{1}, 3); err == nil {
		t.Error("expected error for class index out of range")
	}
	only := New(nn.Classification, []string{"a", "b"})
	_ = only.AddScores([]float64{0.9, 0.1}, 0)
	if only.AUC("a") != 0 || only.MacroAUC() != 0 {
		t.Error("AUC without negatives must be 0")
	}
	if New(nn.Classification, nil).LogLoss() != 0 {
		t.Error("LogLoss without scores must be 0")
	}
}

func TestAUCTiesAndNormalization(t *testing.T) {
	e := New(nn.Classification, []string{"a", "b"})
	// identical scores: AUC must be 0.5, sigmoid outputs are normalized
	_ = e.AddScores([]float64{0.3, 0.3}, 0)
	_ = e.AddScores([]float64{0.3, 0.3}, 1)
	near(t, "AUC with ties", e.AUC("b"), 0.5)
	near(t, "LogLoss of uniform", e.LogLoss(), math.Log(2))
	zero := New(nn.Classification, []string{"a", "b"})
	_ = zero.AddScores([]float64{0, 0}, 0)
	near(t, "LogLoss of zero outputs", zero.LogLoss(), math.Log(2))
}

func TestEvaluateRecordsScores(t *testing.T) {
	set := learn.NewSet(nn.Classification)
	_ = set.AddClass("a")
	_ = set.AddClass("b")
	_ = set.Add(learn.NewClassificationSample([]float64{1, 0}, "a"))
	_ = set.Add(learn.NewClassificationSample([]float64{0, 1}, "b"))
	e, err := Evaluate(identity(t, 2, set.Labels), set, 0)
	if err != nil {
		t.Fatal(err)
	}
	near(t, "AUC", e.MacroAUC(), 1)
	var buf bytes.Buffer
	if err := e.WriteSummary(&buf, "a"); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "ROC-AUC:             1.000") {
		t.Errorf("summary misses AUC:\n%s", buf.String())
	}
	b, _ := json.Marshal(e)
	if strings.Contains(string(b), "scores") {
		t.Error("scores must not be serialized")
	}
	var back Evaluation
	if err := json.Unmarshal(b, &back); err != nil || back.Labels[1] != "b" {
		t.Fatalf("round trip: %v %+v", err, back)
	}
}
