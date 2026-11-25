// Package explain measures which inputs a trained model relies on.
//
// Permutation feature importance shuffles the values of one feature across
// the samples, which breaks its relation to the target while keeping its
// distribution, and measures how much the model gets worse. A feature the
// model ignores costs nothing when shuffled; an important one costs a lot.
// It works for any model and needs no retraining, but it should be
// computed on data the model was not trained on.
//
//	imp, err := explain.Permutation(network, testSet, explain.Config{})
//	for _, f := range imp {
//		fmt.Printf("%-10s %.3f ± %.3f\n", f.Feature, f.Mean, f.Std)
//	}
package explain

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

// Importance is the permutation importance of one feature.
type Importance struct {
	// Feature is the name of the feature (or its index if it has no name).
	Feature string `json:"feature"`
	// Index is the position of the feature in the input.
	Index int `json:"index"`
	// Mean is the average loss of quality when the feature is shuffled, in
	// units of the metric and oriented so that larger means more important.
	// Values around zero mean the model does not use the feature.
	Mean float64 `json:"mean"`
	// Std is the standard deviation of the losses over the repeats.
	Std float64 `json:"std"`
	// Drops holds the loss of every repeat.
	Drops []float64 `json:"drops"`
}

// Config configures the importance computation. The zero value is usable.
type Config struct {
	// Metric measures the quality. Default: evaluation.Accuracy for
	// classification, evaluation.RMSE for regression.
	Metric evaluation.Metric
	// Repeats is the number of shuffles per feature. Default 5.
	Repeats int
	// RegressionThreshold is passed to the evaluation.
	RegressionThreshold float64
	// Rand drives the shuffles. Nil uses a randomly seeded source.
	Rand *rand.Rand
	// Workers is the number of features evaluated at the same time by
	// Permutation and model.Model.Importance. Zero uses GOMAXPROCS. The
	// result does not depend on it. Permute always works sequentially.
	Workers int
}

func (c Config) withDefaults(task nn.Task) Config {
	if c.Metric.Value == nil {
		c.Metric = evaluation.Accuracy
		if task == nn.Regression {
			c.Metric = evaluation.RMSE
		}
	}
	if c.Repeats <= 0 {
		c.Repeats = 5
	}
	if c.Rand == nil {
		c.Rand = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	return c
}

// ScoreFunc evaluates a model on data in which the given feature is
// shuffled: row i takes the value of that feature from row order[i]. It is
// called with feature -1 and a nil order for the unshuffled baseline.
type ScoreFunc func(feature int, order []int) (float64, error)

// Permute computes the permutation importance of every feature from a
// score function, for models that are not plain networks on a learn.Set
// (see model.Model.Importance). features names the features, rows is the
// number of samples. The result is sorted from most to least important.
// The features are evaluated one after another.
func Permute(features []string, rows int, score ScoreFunc, metric evaluation.Metric, cfg Config) ([]Importance, error) {
	cfg.Workers = 1
	return PermuteParallel(features, rows, func() ScoreFunc { return score }, metric, cfg)
}

// PermuteParallel is Permute with several features evaluated at the same
// time (Config.Workers). newScore is called once per worker and must return
// a score function that can run concurrently with the others, e.g. one
// working on its own copy of the data. The result does not depend on the
// number of workers.
func PermuteParallel(features []string, rows int, newScore func() ScoreFunc, metric evaluation.Metric, cfg Config) ([]Importance, error) {
	if len(features) == 0 || rows < 2 {
		return nil, errors.New("explain: need at least one feature and two samples")
	}
	if metric.Value == nil {
		return nil, errors.New("explain: no metric")
	}
	if cfg.Repeats <= 0 {
		cfg.Repeats = 5
	}
	if cfg.Rand == nil {
		cfg.Rand = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	workers := cfg.Workers
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	workers = min(workers, len(features))
	scores := make([]ScoreFunc, workers)
	for w := range scores {
		scores[w] = newScore()
	}
	base, err := scores[0](-1, nil)
	if err != nil {
		return nil, err
	}
	// one seed per feature, drawn in order, so the shuffles do not depend
	// on which worker handles a feature
	seeds := make([]uint64, len(features))
	for j := range seeds {
		seeds[j] = cfg.Rand.Uint64()
	}
	out := make([]Importance, len(features))
	errs := make([]error, len(features))
	var wg sync.WaitGroup
	var next atomic.Int64
	for w := range workers {
		wg.Add(1)
		go func(score ScoreFunc) {
			defer wg.Done()
			order := make([]int, rows)
			for {
				j := int(next.Add(1) - 1)
				if j >= len(features) {
					return
				}
				out[j], errs[j] = importance(features[j], j, score, order, base, metric, cfg.Repeats, seeds[j])
			}
		}(scores[w])
	}
	wg.Wait()
	for j, err := range errs {
		if err != nil {
			return nil, fmt.Errorf("explain: feature %q: %w", features[j], err)
		}
	}
	slices.SortStableFunc(out, func(a, b Importance) int {
		switch {
		case a.Mean > b.Mean:
			return -1
		case a.Mean < b.Mean:
			return 1
		}
		return 0
	})
	return out, nil
}

// importance shuffles one feature Repeats times and returns its importance.
func importance(name string, j int, score ScoreFunc, order []int, base float64, metric evaluation.Metric, repeats int, seed uint64) (Importance, error) {
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	imp := Importance{Feature: name, Index: j, Drops: make([]float64, repeats)}
	for r := range repeats {
		for i := range order {
			order[i] = i
		}
		rng.Shuffle(len(order), func(a, b int) { order[a], order[b] = order[b], order[a] })
		s, err := score(j, order)
		if err != nil {
			return imp, err
		}
		drop := base - s
		if !metric.Higher {
			drop = s - base
		}
		imp.Drops[r] = drop
		imp.Mean += drop
	}
	imp.Mean /= float64(repeats)
	if repeats > 1 {
		for _, d := range imp.Drops {
			imp.Std += (d - imp.Mean) * (d - imp.Mean)
		}
		imp.Std = math.Sqrt(imp.Std / float64(repeats-1))
	}
	return imp, nil
}

// Permutation computes the permutation importance of every input of a
// network on a data set, most important first. Feature names come from
// set.Features. Use data the network was not trained on.
func Permutation(n *nn.Network, set *learn.Set, cfg Config) ([]Importance, error) {
	if n == nil || set == nil {
		return nil, errors.New("explain: nil network or set")
	}
	if set.InputSize() != n.InputSize() {
		return nil, fmt.Errorf("%w: set has %d features, network expects %d", nn.ErrDimension, set.InputSize(), n.InputSize())
	}
	cfg = cfg.withDefaults(set.Task)
	features := make([]string, set.InputSize())
	for j := range features {
		features[j] = fmt.Sprintf("x%d", j+1)
		if j < len(set.Features) && set.Features[j] != "" {
			features[j] = set.Features[j]
		}
	}
	rows := set.Len()
	// every worker shuffles the columns of its own copy of the vectors
	newScore := func() ScoreFunc {
		shuffled := &learn.Set{Task: set.Task, Labels: set.Labels, Samples: make([]*learn.Sample, rows)}
		for i, s := range set.Samples {
			c := *s
			c.Vector = slices.Clone(s.Vector)
			shuffled.Samples[i] = &c
		}
		column := make([]float64, rows)
		return func(feature int, order []int) (float64, error) {
			if feature >= 0 {
				for i, s := range set.Samples {
					column[i] = s.Vector[feature]
				}
				for i, s := range shuffled.Samples {
					s.Vector[feature] = column[order[i]]
				}
				defer func() {
					for i, s := range shuffled.Samples {
						s.Vector[feature] = column[i]
					}
				}()
			}
			e, err := evaluation.Evaluate(n, shuffled, cfg.RegressionThreshold)
			if err != nil {
				return 0, err
			}
			return cfg.Metric.Value(e), nil
		}
	}
	return PermuteParallel(features, rows, newScore, cfg.Metric, cfg)
}
