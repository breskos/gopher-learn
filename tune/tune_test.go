package tune

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

// rings returns two concentric rings, which a network with too few hidden
// neurons cannot separate.
func rings(n int, seed uint64) *learn.Set {
	rng := rand.New(rand.NewPCG(seed, seed))
	s := learn.NewSet(nn.Classification)
	_ = s.AddClass("inner")
	_ = s.AddClass("outer")
	for i := range n {
		label, r := "inner", 0.5
		if i%2 == 1 {
			label, r = "outer", 1.5
		}
		a := rng.Float64() * 2 * math.Pi
		r += rng.NormFloat64() * 0.15
		_ = s.Add(learn.NewClassificationSample([]float64{r * math.Cos(a), r * math.Sin(a)}, label))
	}
	return s
}

func TestSpace(t *testing.T) {
	s := Space{Hidden: [][]int{{4}, {8}}, LearningRate: []float64{0.1, 0.01, 0.001}, LayerNorm: []bool{false, true}}
	if s.Size() != 12 {
		t.Fatalf("size %d", s.Size())
	}
	grid := s.Grid()
	if len(grid) != 12 {
		t.Fatalf("%d candidates", len(grid))
	}
	seen := map[string]bool{}
	for _, c := range grid {
		if c.HiddenActivation != nn.Tanh || c.BatchSize != 32 || c.Dropout != 0 {
			t.Fatalf("defaults not applied: %+v", c)
		}
		seen[c.String()] = true
	}
	if len(seen) != 12 {
		t.Fatalf("%d distinct candidates", len(seen))
	}
	// candidates own their hidden slices
	grid[0].Hidden[0] = 99
	if s.Hidden[0][0] != 4 {
		t.Fatal("grid shares the hidden sizes with the space")
	}

	r := s.Random(5, rand.New(rand.NewPCG(1, 1)))
	if len(r) != 5 {
		t.Fatalf("%d random candidates", len(r))
	}
	distinct := map[string]bool{}
	for _, c := range r {
		distinct[c.String()] = true
	}
	if len(distinct) != 5 {
		t.Fatal("random candidates repeat")
	}
	if len(s.Random(100, nil)) != 12 || len(Space{}.Random(3, nil)) != 1 {
		t.Fatal("random larger than the space")
	}
	c := Candidate{Hidden: []int{8, 4}, HiddenActivation: nn.ReLU, LearningRate: 0.01, BatchSize: 16, Dropout: 0.2, WeightDecay: 1e-4, LayerNorm: true}
	if got := c.String(); got != "hidden=[8 4] relu lr=0.01 batch=16 dropout=0.2 decay=0.0001 layer-norm" {
		t.Fatalf("String: %q", got)
	}
}

func TestSearchFindsCapacity(t *testing.T) {
	data := rings(400, 1)
	candidates := Space{
		Hidden:       [][]int{{1}, {16}},
		LearningRate: []float64{0.03},
	}.Grid()
	var calls int
	cfg := Config{Folds: 4, Epochs: 150, Seed: 7, OnTrial: func(Trial) { calls++ }}
	res, err := Search(context.Background(), data, candidates, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, tr := range res.Ranked() {
		t.Logf("%v: %.3f ± %.3f, epochs %v", tr.Candidate, tr.Mean, tr.Std, tr.Epochs)
	}
	best := res.Best()
	if best.Candidate.Hidden[0] != 16 || best.Mean < 0.95 {
		t.Fatalf("best %v with %.3f", best.Candidate, best.Mean)
	}
	if res.Trials[0].Mean > 0.9 {
		t.Fatalf("one hidden neuron separates rings with %.3f", res.Trials[0].Mean)
	}
	if calls != 2 || res.Metric != "accuracy" || !res.Higher || len(best.Scores) != 4 || best.Std < 0 {
		t.Fatalf("result %+v, %d calls", res, calls)
	}
	if r := res.Ranked(); r[0].Candidate.String() != best.Candidate.String() {
		t.Fatal("ranking")
	}

	// the result does not depend on the number of workers
	cfg.OnTrial = nil
	cfg.Workers = 1
	seq, err := Search(context.Background(), data, candidates, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := range res.Trials {
		if !slices.Equal(seq.Trials[i].Scores, res.Trials[i].Scores) {
			t.Fatalf("trial %d: %v with 1 worker, %v in parallel", i, seq.Trials[i].Scores, res.Trials[i].Scores)
		}
	}

	n, sc, err := Fit(context.Background(), data, best.Candidate, Config{Epochs: 150, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	scaled, _ := sc.TransformSet(rings(200, 2))
	e, _ := evaluation.Evaluate(n, scaled, 0)
	if e.CorrectRatio() < 0.95 {
		t.Fatalf("final model: accuracy %.3f on new data", e.CorrectRatio())
	}
}

func TestSearchRegression(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 3))
	data := learn.NewSet(nn.Regression)
	for range 300 {
		x := rng.Float64()*4 - 2
		_ = data.Add(learn.NewRegressionSample([]float64{x}, math.Sin(2*x)))
	}
	candidates := []Candidate{
		{Hidden: []int{16}, HiddenActivation: nn.Tanh, LearningRate: 0.00001, BatchSize: 32}, // learns nothing
		{Hidden: []int{16}, HiddenActivation: nn.Tanh, LearningRate: 0.01, BatchSize: 16},
	}
	res, err := Search(context.Background(), data, candidates, Config{Folds: 3, Epochs: 200, Seed: 1, NoScaling: true, Patience: -1})
	if err != nil {
		t.Fatal(err)
	}
	if res.Metric != "rmse" || res.Higher || res.BestIndex != 1 {
		t.Fatalf("metric %s, best %d: %+v", res.Metric, res.BestIndex, res.Trials)
	}
	if res.Best().Mean > 0.2 {
		t.Fatalf("RMSE %.3f", res.Best().Mean)
	}
	for _, e := range res.Best().Epochs {
		if e != 200 {
			t.Fatalf("early stopping should be off: %v", res.Best().Epochs)
		}
	}
	n, sc, err := Fit(context.Background(), data, candidates[1], Config{NoScaling: true, Seed: 1})
	if err != nil || sc != nil || n.OutputSize() != 1 {
		t.Fatalf("Fit: %v, scaler %v", err, sc)
	}
}

func TestSearchErrors(t *testing.T) {
	data := rings(40, 1)
	ok := []Candidate{{Hidden: []int{4}, LearningRate: 0.01, BatchSize: 8}}
	ctx := context.Background()
	if _, err := Search(ctx, nil, ok, Config{}); err == nil {
		t.Error("no data: expected error")
	}
	if _, err := Search(ctx, data, nil, Config{}); err == nil {
		t.Error("no candidates: expected error")
	}
	if _, err := Search(ctx, data, ok, Config{Folds: 1}); err == nil {
		t.Error("one fold: expected error")
	}
	if _, err := Search(ctx, data, ok, Config{Folds: 100}); err == nil {
		t.Error("more folds than samples: expected error")
	}
	bad := []Candidate{{Hidden: []int{-1}, LearningRate: 0.01}}
	if _, err := Search(ctx, data, bad, Config{Epochs: 1}); err == nil {
		t.Error("invalid candidate: expected error")
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := Search(cancelled, data, ok, Config{}); !errors.Is(err, context.Canceled) {
		t.Errorf("want context.Canceled, got %v", err)
	}
	if _, _, err := Fit(ctx, nil, ok[0], Config{}); err == nil {
		t.Error("Fit without data: expected error")
	}
	if _, _, err := Fit(ctx, data, bad[0], Config{}); err == nil {
		t.Error("Fit with an invalid candidate: expected error")
	}
	// a constant feature cannot break the scaler, but a set without samples in a fold can
	if !better(true, 1, math.NaN()) || better(true, math.NaN(), 1) || !better(false, 1, 2) {
		t.Error("better")
	}
}
