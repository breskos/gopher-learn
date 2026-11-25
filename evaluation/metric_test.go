package evaluation

import (
	"testing"

	"github.com/breskos/gopher-learn/nn"
)

func TestMetricType(t *testing.T) {
	e := New(nn.Classification, []string{"a", "b"})
	e.Add("a", "a")
	e.Add("b", "a")
	if v := Accuracy.Value(e); v != 0.5 {
		t.Fatalf("accuracy %v", v)
	}
	if !Accuracy.Better(0.9, 0.8) || Accuracy.Better(0.8, 0.9) || !RMSE.Better(0.1, 0.2) || RMSE.Better(0.2, 0.1) {
		t.Fatal("Better")
	}
	for _, m := range metrics {
		got, err := MetricByName(m.Name)
		if err != nil || got.Name != m.Name || got.Higher != m.Higher {
			t.Fatalf("%s: %v", m.Name, err)
		}
		_ = got.Value(e) // every metric works on any evaluation
	}
	if _, err := MetricByName("nope"); err == nil {
		t.Fatal("expected error")
	}
	r := New(nn.Regression, nil)
	r.AddRegression(1, 2)
	if MAE.Value(r) != 1 || RMSE.Value(r) != 1 {
		t.Fatal("regression metrics")
	}
}
