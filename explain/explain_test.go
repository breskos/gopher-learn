package explain

import (
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

// data returns samples whose class depends strongly on x1, weakly on x2 and
// not at all on x3.
func data(n int, seed uint64) *learn.Set {
	rng := rand.New(rand.NewPCG(seed, seed))
	s := learn.NewSet(nn.Classification)
	s.Features = []string{"strong", "weak"} // the third feature has no name
	_ = s.AddClass("a")
	_ = s.AddClass("b")
	for range n {
		x := []float64{rng.NormFloat64(), rng.NormFloat64(), rng.NormFloat64()}
		label := "a"
		if 2*x[0]+0.5*x[1] > 0 {
			label = "b"
		}
		_ = s.Add(learn.NewClassificationSample(x, label))
	}
	return s
}

func TestPermutation(t *testing.T) {
	train, test := data(2000, 1), data(1000, 2)
	n, _ := nn.New(nn.Config{Inputs: 3, Hidden: []int{8}, Labels: train.Labels, HiddenActivation: nn.Tanh, OutputActivation: nn.Softmax, Rand: rand.New(rand.NewPCG(1, 1))})
	if _, err := nn.Fit(context.Background(), n, train.Data(), nn.Data{}, nn.FitConfig{Epochs: 30, Rand: rand.New(rand.NewPCG(2, 2))}); err != nil {
		t.Fatal(err)
	}
	before := slices.Clone(test.Samples[0].Vector)
	imp, err := Permutation(n, test, Config{Repeats: 4, Rand: rand.New(rand.NewPCG(3, 3))})
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range imp {
		t.Logf("%-6s %.3f ± %.3f", f.Feature, f.Mean, f.Std)
	}
	if imp[0].Feature != "strong" || imp[1].Feature != "weak" || imp[2].Feature != "x3" || imp[2].Index != 2 {
		t.Fatalf("order %v", imp)
	}
	if imp[0].Mean < 0.3 || imp[1].Mean < 0.02 || math.Abs(imp[2].Mean) > 0.01 || len(imp[0].Drops) != 4 {
		t.Fatalf("importances %+v", imp)
	}
	if !slices.Equal(before, test.Samples[0].Vector) {
		t.Fatal("the input set was changed")
	}

	// a lower-is-better metric is oriented the same way
	imp, err = Permutation(n, test, Config{Metric: evaluation.LogLoss, Repeats: 2, Rand: rand.New(rand.NewPCG(3, 3))})
	if err != nil || imp[0].Feature != "strong" || imp[0].Mean <= 0 {
		t.Fatalf("log loss: %+v, %v", imp, err)
	}
}

func TestPermuteErrors(t *testing.T) {
	ok := func(int, []int) (float64, error) { return 1, nil }
	if _, err := Permute(nil, 10, ok, evaluation.Accuracy, Config{}); err == nil {
		t.Error("no features: expected error")
	}
	if _, err := Permute([]string{"a"}, 1, ok, evaluation.Accuracy, Config{}); err == nil {
		t.Error("one row: expected error")
	}
	if _, err := Permute([]string{"a"}, 3, ok, evaluation.Metric{}, Config{}); err == nil {
		t.Error("no metric: expected error")
	}
	boom := errors.New("boom")
	if _, err := Permute([]string{"a"}, 3, func(f int, _ []int) (float64, error) { return 0, boom }, evaluation.Accuracy, Config{}); !errors.Is(err, boom) {
		t.Errorf("baseline error: %v", err)
	}
	if _, err := Permute([]string{"a"}, 3, func(f int, _ []int) (float64, error) {
		if f >= 0 {
			return 0, boom
		}
		return 1, nil
	}, evaluation.Accuracy, Config{}); !errors.Is(err, boom) {
		t.Errorf("feature error: %v", err)
	}
	imp, err := Permute([]string{"a", "b"}, 3, ok, evaluation.RMSE, Config{Repeats: 1})
	if err != nil || imp[0].Mean != 0 || imp[0].Std != 0 {
		t.Fatalf("constant score: %+v, %v", imp, err)
	}

	n, _ := nn.New(nn.Config{Inputs: 2, Outputs: 1})
	if _, err := Permutation(nil, data(5, 1), Config{}); err == nil {
		t.Error("nil network: expected error")
	}
	if _, err := Permutation(n, data(5, 1), Config{}); !errors.Is(err, nn.ErrDimension) {
		t.Errorf("dimension: %v", err)
	}
	r := learn.NewSet(nn.Regression)
	_ = r.Add(learn.NewRegressionSample([]float64{1, 2}, 1))
	_ = r.Add(learn.NewRegressionSample([]float64{2, 1}, 2))
	imp, err = Permutation(n, r, Config{})
	if err != nil || len(imp) != 2 || imp[0].Feature != "x1" && imp[0].Feature != "x2" {
		t.Fatalf("regression default metric: %+v, %v", imp, err)
	}
	bad := learn.NewSet(nn.Classification)
	bad.Labels = []string{"a"}
	bad.Samples = []*learn.Sample{{Vector: []float64{1, 2}, Output: []float64{1, 0}}, {Vector: []float64{1, 2}, Output: []float64{1, 0}}}
	if _, err := Permutation(n, bad, Config{}); err == nil {
		t.Error("evaluation error: expected error")
	}
}
