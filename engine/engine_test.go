package engine

import (
	"context"
	"errors"
	"log/slog"
	"math/rand/v2"
	"testing"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

func blobs(t *testing.T, n int) *learn.Set {
	t.Helper()
	rng := rand.New(rand.NewPCG(7, 7))
	s := learn.NewSet(nn.Classification)
	_ = s.AddClass("left")
	_ = s.AddClass("right")
	for i := range n {
		label, cx := "left", -1.0
		if i%2 == 1 {
			label, cx = "right", 1.0
		}
		v := []float64{cx + rng.NormFloat64()*0.3, rng.NormFloat64() * 0.3}
		if err := s.Add(learn.NewClassificationSample(v, label)); err != nil {
			t.Fatal(err)
		}
	}
	return s
}

func TestEngineClassification(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Epochs = 5
	cfg.LearningRate = 0.3
	cfg.Decay = 0.1
	cfg.Seed = 42
	e, err := New(blobs(t, 200), []int{4}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Run(context.Background(), FMeasure); err != nil {
		t.Fatal(err)
	}
	net, eval := e.Winner()
	if net == nil || eval == nil {
		t.Fatal("no winner")
	}
	if f1 := eval.OverallFMeasure(); f1 < 0.9 {
		t.Fatalf("F1 = %.3f, want >= 0.9", f1)
	}
}

func TestEngineRegression(t *testing.T) {
	s := learn.NewSet(nn.Regression)
	rng := rand.New(rand.NewPCG(3, 3))
	for range 200 {
		x := rng.Float64()
		_ = s.Add(learn.NewRegressionSample([]float64{x}, 0.2+0.6*x))
	}
	cfg := DefaultConfig()
	cfg.Epochs = 10
	cfg.LearningRate = 0.5
	cfg.Decay = 0.1
	cfg.Seed = 1
	cfg.RegressionThreshold = 0.05
	e, err := New(s, []int{4}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Run(context.Background(), Distance); err != nil {
		t.Fatal(err)
	}
	_, eval := e.Winner()
	if r := eval.CorrectRatio(); r < 0.9 {
		t.Fatalf("correct ratio = %.3f, want >= 0.9", r)
	}
}

func TestEngineCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	e, err := New(blobs(t, 20), []int{2}, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Run(ctx, Distance); !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
}

func TestConfigValidate(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Decay = 0
	if _, err := New(blobs(t, 4), nil, cfg); err == nil {
		t.Fatal("expected error for zero decay")
	}
}

func TestRounds(t *testing.T) {
	cfg := DefaultConfig()
	cfg.LearningRate, cfg.Decay = 0.3, 0.1
	if r := cfg.rates(); len(r) != 3 || r[0] != 0.3 {
		t.Fatalf("rates %v", r)
	}

	// every round ends with an evaluation; the winner comes from one of them
	cfg.Epochs, cfg.Seed = 2, 9
	var rounds []float64
	cfg.Logger = slog.New(handlerFunc(func(r slog.Record) {
		if r.Message == "round" {
			r.Attrs(func(a slog.Attr) bool {
				if a.Key == "learning_rate" {
					rounds = append(rounds, a.Value.Float64())
				}
				return true
			})
		}
	}))
	e, err := New(blobs(t, 40), []int{2}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Run(context.Background(), Accuracy); err != nil {
		t.Fatal(err)
	}
	if len(rounds) != 3 || rounds[2] > 0.11 || rounds[2] < 0.09 {
		t.Fatalf("round learning rates %v", rounds)
	}
}

func TestEngineAdam(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Epochs, cfg.LearningRate, cfg.Decay, cfg.Seed = 10, 0.01, 0.005, 3
	cfg.Optimizer = nn.DefaultAdam
	cfg.WeightDecay = 1e-4
	cfg.BatchSize = 16
	cfg.HiddenActivation, cfg.OutputActivation = nn.Tanh, nn.Softmax
	e, err := New(blobs(t, 200), []int{8}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Run(context.Background(), FMeasure); err != nil {
		t.Fatal(err)
	}
	if _, eval := e.Winner(); eval.OverallFMeasure() < 0.9 {
		t.Fatalf("F1 = %.3f", eval.OverallFMeasure())
	}
}

func TestEngineDiverges(t *testing.T) {
	s := learn.NewSet(nn.Regression)
	for i := range 20 {
		_ = s.Add(learn.NewRegressionSample([]float64{float64(i)}, float64(10*i)))
	}
	cfg := DefaultConfig()
	cfg.LearningRate, cfg.Decay, cfg.Epochs, cfg.Seed = 50, 25, 50, 1
	cfg.HiddenActivation, cfg.OutputActivation = nn.Linear, nn.Linear
	var warned bool
	cfg.Logger = slog.New(handlerFunc(func(r slog.Record) { warned = warned || r.Level == slog.LevelWarn }))
	e, err := New(s, []int{4}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Run(context.Background(), Distance); err != nil {
		t.Fatal(err)
	}
	if !warned {
		t.Fatal("expected a warning about divergence")
	}
}

// handlerFunc is a slog handler that passes every record to a function.
type handlerFunc func(slog.Record)

func (h handlerFunc) Enabled(context.Context, slog.Level) bool      { return true }
func (h handlerFunc) Handle(_ context.Context, r slog.Record) error { h(r); return nil }
func (h handlerFunc) WithAttrs([]slog.Attr) slog.Handler            { return h }
func (h handlerFunc) WithGroup(string) slog.Handler                 { return h }
