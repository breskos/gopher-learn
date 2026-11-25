package engine

import (
	"context"
	"errors"
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
