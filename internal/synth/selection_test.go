package synth_test

import (
	"context"
	"math/rand/v2"
	"testing"

	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/explain"
	"github.com/breskos/gopher-learn/internal/synth"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/tune"
)

func dataset(t *testing.T, path string, rows int) *learn.Set {
	t.Helper()
	d, err := synth.ByPath(path)
	if err != nil {
		t.Fatal(err)
	}
	s, err := d.Set(rows, 42)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// TestTuneSpirals checks that a random search finds a configuration that
// separates the spirals and that the search ranks it above weak ones.
func TestTuneSpirals(t *testing.T) {
	if testing.Short() {
		t.Skip("quality tests train real models")
	}
	data := dataset(t, "classification/spirals.csv", 1500)
	space := tune.Space{
		Hidden:       [][]int{{4}, {32}, {32, 32}},
		LearningRate: []float64{0.001, 0.01, 0.03},
	}
	res, err := tune.Search(context.Background(), data, space.Random(6, rand.New(rand.NewPCG(1, 1))), tune.Config{Folds: 3, Epochs: 150, Seed: 3})
	if err != nil {
		t.Fatal(err)
	}
	ranked := res.Ranked()
	for _, tr := range ranked {
		t.Logf("%-40v %.3f ± %.3f", tr.Candidate, tr.Mean, tr.Std)
	}
	if best := res.Best(); best.Mean < 0.95 || len(best.Candidate.Hidden) == 1 && best.Candidate.Hidden[0] == 4 {
		t.Errorf("best candidate %v with %.3f", best.Candidate, best.Mean)
	}
	if worst := ranked[len(ranked)-1]; worst.Mean > 0.9 {
		t.Errorf("the search does not separate good from bad candidates: worst %.3f", worst.Mean)
	}
}

// TestImportanceFriedman checks permutation importance on Friedman #1,
// where x1 to x5 determine the target and x6 to x10 are irrelevant.
func TestImportanceFriedman(t *testing.T) {
	if testing.Short() {
		t.Skip("quality tests train real models")
	}
	data := dataset(t, "regression/friedman1.csv", 6000)
	train, test, err := data.Split(0.7, rand.New(rand.NewPCG(1, 1)))
	if err != nil {
		t.Fatal(err)
	}
	c := tune.Candidate{Hidden: []int{32, 16}, HiddenActivation: nn.Tanh, LearningRate: 0.01, BatchSize: 32}
	n, sc, err := tune.Fit(context.Background(), train, c, tune.Config{Epochs: 200, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	scaled, err := sc.TransformSet(test)
	if err != nil {
		t.Fatal(err)
	}
	e, _ := evaluation.Evaluate(n, scaled, 0)
	imp, err := explain.Permutation(n, scaled, explain.Config{Repeats: 3, Rand: rand.New(rand.NewPCG(2, 2))})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("R² %.3f", e.R2())
	relevant := map[string]bool{"x1": true, "x2": true, "x3": true, "x4": true, "x5": true}
	for i, f := range imp {
		t.Logf("%-4s %.4f ± %.4f", f.Feature, f.Mean, f.Std)
		if relevant[f.Feature] != (i < 5) {
			t.Errorf("%s ranks %d", f.Feature, i+1)
		}
	}
	if imp[5].Mean > 0.1*imp[4].Mean {
		t.Errorf("irrelevant %s at %.4f is close to relevant %s at %.4f", imp[5].Feature, imp[5].Mean, imp[4].Feature, imp[4].Mean)
	}
}
