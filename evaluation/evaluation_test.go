package evaluation

import (
	"bytes"
	"math"
	"strings"
	"testing"

	"github.com/breskos/gopher-learn/nn"
)

func sample() *Evaluation {
	e := New(nn.Classification, []string{"a", "b"})
	// actual a: 3x a, 1x b; actual b: 2x b, 2x a
	for _, p := range [][2]string{{"a", "a"}, {"a", "a"}, {"a", "a"}, {"a", "b"}, {"b", "b"}, {"b", "b"}, {"b", "a"}, {"b", "a"}} {
		e.Add(p[0], p[1])
	}
	return e
}

func TestCounts(t *testing.T) {
	e := sample()
	got := [4]int{e.TruePositives("a"), e.FalsePositives("a"), e.FalseNegatives("a"), e.TrueNegatives("a")}
	if want := [4]int{3, 2, 1, 2}; got != want {
		t.Fatalf("TP/FP/FN/TN = %v, want %v", got, want)
	}
	if e.Correct != 5 || e.Wrong != 3 {
		t.Fatalf("correct/wrong = %d/%d", e.Correct, e.Wrong)
	}
}

func TestMetrics(t *testing.T) {
	e := sample()
	near := func(name string, got, want float64) {
		t.Helper()
		if math.Abs(got-want) > 1e-9 {
			t.Errorf("%s = %v, want %v", name, got, want)
		}
	}
	near("precision", e.Precision("a"), 3.0/5)
	near("recall", e.Recall("a"), 3.0/4)
	near("f1", e.FMeasure("a"), 6.0/9)
	near("accuracy", e.Accuracy("a"), 5.0/8)
	near("balanced", e.BalancedAccuracy("a"), (3.0/4+2.0/4)/2)
	near("ratio", e.CorrectRatio(), 5.0/8)
}

func TestEmpty(t *testing.T) {
	e := New(nn.Classification, nil)
	if e.CorrectRatio() != 0 || e.OverallFMeasure() != 0 {
		t.Fatal("empty evaluation must not produce NaN")
	}
}

func TestReport(t *testing.T) {
	var buf bytes.Buffer
	e := sample()
	if err := e.WriteConfusionMatrix(&buf); err != nil {
		t.Fatal(err)
	}
	if err := e.WriteSummaries(&buf); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "summary for class b") {
		t.Fatalf("unexpected report:\n%s", buf.String())
	}
}
