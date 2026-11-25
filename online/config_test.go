package online

import (
	"context"
	"errors"
	"math/rand/v2"
	"testing"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

func TestConfigValidate(t *testing.T) {
	cases := map[string]func(*Config){
		"split":      func(c *Config) { c.TrainingSplit = 0 },
		"min epochs": func(c *Config) { c.MinEpochs = 0 },
		"epochs":     func(c *Config) { c.MaxEpochs = c.MinEpochs - 1 },
		"min lr":     func(c *Config) { c.MinLearningRate = 0 },
		"lr":         func(c *Config) { c.MaxLearningRate = c.MinLearningRate / 2 },
		"min data":   func(c *Config) { c.MinimumDataPoints = 1 },
	}
	for name, mutate := range cases {
		cfg := DefaultConfig()
		mutate(&cfg)
		if _, err := New(learn.NewSet(nn.Regression), 2, nil, cfg); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
}

func TestNewErrors(t *testing.T) {
	if _, err := New(nil, 2, nil, DefaultConfig()); err == nil {
		t.Error("expected error for nil data")
	}
	// classification set without classes has no outputs
	if _, err := New(learn.NewSet(nn.Classification), 2, nil, DefaultConfig()); err == nil {
		t.Error("expected error for a set without classes")
	}
}

func TestInjectErrors(t *testing.T) {
	set := learn.NewSet(nn.Classification)
	_ = set.AddClass("a")
	o, err := New(set, 2, nil, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := o.Inject(learn.NewClassificationSample([]float64{1, 2}, "unknown"), false); !errors.Is(err, learn.ErrUnknownClass) {
		t.Errorf("want ErrUnknownClass, got %v", err)
	}
	if err := o.Inject(learn.NewClassificationSample([]float64{1, 2, 3}, "a"), true); !errors.Is(err, nn.ErrDimension) {
		t.Errorf("want ErrDimension, got %v", err)
	}
	// force re-adds an existing sample
	s := learn.NewClassificationSample([]float64{1, 2}, "a")
	if err := o.Inject(s, false); err != nil {
		t.Fatal(err)
	}
	if err := o.Inject(learn.NewClassificationSample([]float64{1, 2}, "a"), true); err != nil {
		t.Fatalf("forced inject failed: %v", err)
	}
}

func TestRegressionAndInit(t *testing.T) {
	set := learn.NewSet(nn.Regression)
	rng := rand.New(rand.NewPCG(2, 2))
	for range 60 {
		x := rng.Float64()
		_ = set.Add(learn.NewRegressionSample([]float64{x}, 0.2+0.6*x))
	}
	cfg := DefaultConfig()
	cfg.Seed = 11
	cfg.RegressionThreshold = 0.1
	cfg.TargetScore = 0.8
	cfg.MaxInitLoops = 10
	o, err := New(set, 1, []int{4}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	score, err := o.Init(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if score < cfg.TargetScore {
		t.Fatalf("Init score %.3f below target %.3f", score, cfg.TargetScore)
	}
	if o.LastEvaluation == nil || o.LastEvaluation.Task != nn.Regression {
		t.Fatal("LastEvaluation not set")
	}
}

func TestInitWithoutData(t *testing.T) {
	set := learn.NewSet(nn.Regression)
	o, _ := New(set, 1, nil, DefaultConfig())
	if _, err := o.Init(context.Background()); !errors.Is(err, ErrNotEnoughData) {
		t.Fatalf("want ErrNotEnoughData, got %v", err)
	}
}

func TestIterateCancelled(t *testing.T) {
	set := learn.NewSet(nn.Regression)
	for i := range 20 {
		_ = set.Add(learn.NewRegressionSample([]float64{float64(i)}, 0.5))
	}
	o, _ := New(set, 1, nil, DefaultConfig())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := o.Iterate(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
}
