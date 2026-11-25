package engine

import (
	"testing"

	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

func evalWith(pairs [][2]string, distance float64) *evaluation.Evaluation {
	e := evaluation.New(nn.Classification, []string{"a", "b"})
	for _, p := range pairs {
		e.Add(p[0], p[1])
	}
	e.TotalDistance = distance
	return e
}

func TestCriterionBetter(t *testing.T) {
	good := evalWith([][2]string{{"a", "a"}, {"b", "b"}, {"a", "a"}, {"b", "b"}}, 0.1)
	bad := evalWith([][2]string{{"a", "b"}, {"b", "b"}, {"a", "a"}, {"b", "a"}}, 0.9)
	for _, c := range []Criterion{Accuracy, BalancedAccuracy, FMeasure, Simple, Distance} {
		t.Run(c.String(), func(t *testing.T) {
			if !c.better(nil, bad) {
				t.Error("anything beats no evaluation")
			}
			if !c.better(bad, good) {
				t.Error("good must beat bad")
			}
			if c.better(good, bad) {
				t.Error("bad must not beat good")
			}
			if c.better(good, good) {
				t.Error("equal must not count as better")
			}
		})
	}
	if Criterion(42).better(bad, good) {
		t.Error("unknown criterion must never report better")
	}
	if Criterion(42).String() != "Criterion(42)" {
		t.Error("unexpected name for invalid criterion")
	}
}

func TestConfigValidateCases(t *testing.T) {
	cases := map[string]func(*Config){
		"tries":  func(c *Config) { c.Tries = 0 },
		"epochs": func(c *Config) { c.Epochs = 0 },
		"split":  func(c *Config) { c.TrainingSplit = 1 },
		"lr":     func(c *Config) { c.LearningRate = 0 },
		"decay":  func(c *Config) { c.Decay = -1 },
	}
	for name, mutate := range cases {
		cfg := DefaultConfig()
		mutate(&cfg)
		if err := cfg.Validate(); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
	if err := DefaultConfig().Validate(); err != nil {
		t.Fatalf("default config invalid: %v", err)
	}
}

func TestNewErrors(t *testing.T) {
	if _, err := New(nil, nil, DefaultConfig()); err == nil {
		t.Error("expected error for nil data")
	}
	one := learn.NewSet(nn.Regression)
	_ = one.Add(learn.NewRegressionSample([]float64{1}, 0.5))
	if _, err := New(one, nil, DefaultConfig()); err == nil {
		t.Error("expected error for a single sample")
	}
	e, err := New(blobs(t, 10), []int{2}, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if e.Config().Epochs != DefaultConfig().Epochs {
		t.Error("Config must return the engine configuration")
	}
	if n, ev := e.Winner(); n != nil || ev != nil {
		t.Error("winner must be nil before Run")
	}
}

func TestBadHiddenLayer(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Decay = 0.4
	e, err := New(blobs(t, 10), []int{0}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Run(t.Context(), Distance); err == nil {
		t.Fatal("expected error for a hidden layer of size 0")
	}
}
