package synth_test

import (
	"context"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/breskos/gopher-learn/engine"
	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/internal/synth"
	"github.com/breskos/gopher-learn/learn"
)

// These tests train real models on the synthetic data sets and check the
// quality on a held-out test set. They guard against regressions in
// training and data handling and take a few seconds; -short skips them.

type qualityCase struct {
	path    string
	rows    int
	hidden  []int
	scale   bool
	cfg     func(*engine.Config)
	measure func(*evaluation.Evaluation) float64
	// min is the lowest acceptable quality. Measured values (seed 1) are
	// noted next to each case; the margin absorbs platform differences
	// in floating point (e.g. fused multiply-add on arm64).
	min float64
}

func accuracy(e *evaluation.Evaluation) float64         { return e.CorrectRatio() }
func balancedAccuracy(e *evaluation.Evaluation) float64 { return e.OverallBalancedAccuracy() }

func baseConfig() engine.Config {
	cfg := engine.DefaultConfig()
	cfg.Seed = 1
	cfg.Epochs = 10
	cfg.LearningRate = 0.3
	cfg.Decay = 0.1
	return cfg
}

// train trains on 80 % of the data and evaluates the winner on the rest.
func train(t *testing.T, c qualityCase) (*evaluation.Evaluation, time.Duration) {
	t.Helper()
	d, err := synth.ByPath(c.path)
	if err != nil {
		t.Fatal(err)
	}
	data, err := d.Set(c.rows, 42)
	if err != nil {
		t.Fatal(err)
	}
	trainSet, testSet, err := data.StratifiedSplit(0.8, rand.New(rand.NewPCG(1, 1)))
	if err != nil {
		t.Fatal(err)
	}
	if c.scale {
		sc, err := learn.FitScaler(trainSet, learn.Standardize)
		if err != nil {
			t.Fatal(err)
		}
		if trainSet, err = sc.TransformSet(trainSet); err != nil {
			t.Fatal(err)
		}
		if testSet, err = sc.TransformSet(testSet); err != nil {
			t.Fatal(err)
		}
	}
	cfg := baseConfig()
	if c.cfg != nil {
		c.cfg(&cfg)
	}
	start := time.Now()
	e, err := engine.New(trainSet, c.hidden, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Run(context.Background(), engine.Distance); err != nil {
		t.Fatal(err)
	}
	net, _ := e.Winner()
	eval, err := evaluation.Evaluate(net, testSet, cfg.RegressionThreshold)
	if err != nil {
		t.Fatal(err)
	}
	return eval, time.Since(start)
}

func TestQuality(t *testing.T) {
	if testing.Short() {
		t.Skip("quality tests train real models")
	}
	cases := []qualityCase{
		{path: "classification/blobs.csv", rows: 2000, hidden: []int{8}, measure: accuracy, min: 0.97},              // measured 1.000
		{path: "classification/imbalanced.csv", rows: 5000, hidden: []int{8}, measure: balancedAccuracy, min: 0.80}, // measured 0.878
		{path: "classification/high-dim.csv", rows: 3000, hidden: []int{16}, measure: accuracy, min: 0.85},          // measured 0.927
		// Spirals need many epochs; tanh with a high learning rate saturates.
		{path: "classification/spirals.csv", rows: 3000, hidden: []int{32},
			cfg:     func(c *engine.Config) { c.Epochs = 60 },
			measure: accuracy, min: 0.90}, // measured 0.995
		{path: "regression/friedman1.csv", rows: 5000, hidden: []int{16}, scale: true,
			cfg:     func(c *engine.Config) { c.RegressionThreshold = 0.05; c.LearningRate, c.Decay = 0.1, 0.02 },
			measure: accuracy, min: 0.50}, // measured 0.602
	}
	for _, c := range cases {
		t.Run(c.path, func(t *testing.T) {
			eval, took := train(t, c)
			got := c.measure(eval)
			t.Logf("%s: %.3f in %v", c.path, got, took.Round(time.Millisecond))
			if got < c.min {
				t.Fatalf("quality %.3f below %.3f", got, c.min)
			}
		})
	}
}

func TestScalingHelpsUnscaled(t *testing.T) {
	if testing.Short() {
		t.Skip("quality tests train real models")
	}
	run := func(scale bool) float64 {
		eval, took := train(t, qualityCase{
			path: "regression/unscaled.csv", rows: 4000, hidden: []int{16}, scale: scale,
			cfg: func(c *engine.Config) { c.RegressionThreshold = 0.05; c.LearningRate, c.Decay = 0.1, 0.02 },
		})
		t.Logf("scale=%v: distance %.3f, correct %.3f in %v", scale, eval.TotalDistance, eval.CorrectRatio(), took.Round(time.Millisecond))
		return eval.TotalDistance
	}
	raw, scaled := run(false), run(true)
	if scaled >= raw {
		t.Fatalf("scaling did not help: distance %.3f scaled vs %.3f raw", scaled, raw)
	}
}
