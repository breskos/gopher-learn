package synth_test

import (
	"context"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/breskos/gopher-learn/cluster"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/linear"
	"github.com/breskos/gopher-learn/pca"
	"github.com/breskos/gopher-learn/tune"
)

func split(t *testing.T, s *learn.Set) (*learn.Set, *learn.Set) {
	t.Helper()
	a, b, err := s.StratifiedSplit(0.7, rand.New(rand.NewPCG(1, 1)))
	if err != nil {
		t.Fatal(err)
	}
	return a, b
}

// TestClassicBaselines checks the classic models on the synthetic data and
// compares them with a neural network where that is informative.
func TestClassicBaselines(t *testing.T) {
	if testing.Short() {
		t.Skip("quality tests train real models")
	}
	t.Run("logistic/blobs", func(t *testing.T) {
		train, test := split(t, dataset(t, "classification/blobs.csv", 4000))
		m, err := linear.FitLogistic(train, linear.LogisticConfig{})
		if err != nil {
			t.Fatal(err)
		}
		e, _ := m.Evaluate(test)
		t.Logf("accuracy %.3f after %d iterations", e.CorrectRatio(), m.Iterations)
		if e.CorrectRatio() < 0.99 {
			t.Errorf("accuracy %.3f", e.CorrectRatio())
		}
	})
	t.Run("logistic/imbalanced", func(t *testing.T) {
		train, test := split(t, dataset(t, "classification/imbalanced.csv", 6000))
		m, err := linear.FitLogistic(train, linear.LogisticConfig{})
		if err != nil {
			t.Fatal(err)
		}
		e, _ := m.Evaluate(test)
		t.Logf("accuracy %.3f, macro F1 %.3f, macro AUC %.3f", e.CorrectRatio(), e.OverallFMeasure(), e.MacroAUC())
		if e.MacroAUC() < 0.9 {
			t.Errorf("macro AUC %.3f", e.MacroAUC())
		}
	})
	t.Run("regression/friedman1", func(t *testing.T) {
		train, test := split(t, dataset(t, "regression/friedman1.csv", 6000))
		m, err := linear.FitRegression(train, linear.RegressionConfig{})
		if err != nil {
			t.Fatal(err)
		}
		e, _ := m.Evaluate(test, 0)
		// the network captures the nonlinear terms the linear model misses
		c := tune.Candidate{Hidden: []int{32, 16}, LearningRate: 0.01, BatchSize: 32}
		n, sc, err := tune.Fit(context.Background(), train, c, tune.Config{Epochs: 200, Seed: 1})
		if err != nil {
			t.Fatal(err)
		}
		scaled, _ := sc.TransformSet(test)
		en, _ := tuneEval(n, scaled)
		t.Logf("R² linear %.3f, network %.3f", e.R2(), en)
		if e.R2() < 0.6 || en < e.R2()+0.1 {
			t.Errorf("R² linear %.3f, network %.3f", e.R2(), en)
		}
	})
	t.Run("kmeans/blobs", func(t *testing.T) {
		data := dataset(t, "classification/blobs.csv", 4000)
		points := data.Data().X
		m, labels, err := cluster.Fit(points, cluster.Config{K: 4, Rand: rand.New(rand.NewPCG(2, 2))})
		if err != nil {
			t.Fatal(err)
		}
		counts := make([][]int, 4)
		for c := range counts {
			counts[c] = make([]int, len(data.Labels))
		}
		for i, l := range labels {
			counts[l][data.Samples[i].Class]++
		}
		correct := 0
		for _, row := range counts {
			correct += slices.Max(row)
		}
		purity := float64(correct) / float64(len(labels))
		sil, _ := cluster.Silhouette(points[:1000], labels[:1000])
		t.Logf("purity %.3f, silhouette %.3f, %d iterations", purity, sil, m.Iterations)
		if purity < 0.97 {
			t.Errorf("purity %.3f", purity)
		}
	})
	t.Run("pca/high-dim", func(t *testing.T) {
		train, test := split(t, dataset(t, "classification/high-dim.csv", 4000))
		p, err := pca.Fit(train.Data().X, pca.Config{Standardize: true, Components: 20})
		if err != nil {
			t.Fatal(err)
		}
		r := p.ExplainedRatio()
		var top float64
		for _, v := range r[:10] {
			top += v
		}
		full, _ := linear.FitLogistic(train, linear.LogisticConfig{L2: 0.01})
		ef, _ := full.Evaluate(test)
		trainZ, _ := p.TransformSet(train)
		testZ, _ := p.TransformSet(test)
		small, _ := linear.FitLogistic(trainZ, linear.LogisticConfig{L2: 0.01})
		es, _ := small.Evaluate(testZ)
		t.Logf("10 of 200 components explain %.3f; accuracy with 200 features %.3f, with 20 components %.3f", top, ef.CorrectRatio(), es.CorrectRatio())
		// x1–x10 and their 20 noisy copies span about 10 strong directions
		if top < 0.15 || r[9] < 1.5*r[10] { // a clear gap after the 10th component
			t.Errorf("explained %v", r)
		}
		if es.CorrectRatio() < ef.CorrectRatio()-0.02 {
			t.Errorf("20 components lose accuracy: %.3f vs %.3f", es.CorrectRatio(), ef.CorrectRatio())
		}
	})
}

func tuneEval(n interface {
	Predict([]float64) ([]float64, error)
}, s *learn.Set) (float64, error) {
	var ssRes, ssTot, mean float64
	for _, sm := range s.Samples {
		mean += sm.Value
	}
	mean /= float64(s.Len())
	for _, sm := range s.Samples {
		out, err := n.Predict(sm.Vector)
		if err != nil {
			return 0, err
		}
		ssRes += (out[0] - sm.Value) * (out[0] - sm.Value)
		ssTot += (sm.Value - mean) * (sm.Value - mean)
	}
	return 1 - ssRes/ssTot, nil
}
