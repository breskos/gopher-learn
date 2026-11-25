// Package tune searches for good hyperparameters with k-fold
// cross-validation.
//
// Describe the values to try in a Space, turn it into candidates with
// Space.Grid (every combination) or Space.Random (a sample of them) and
// evaluate them with Search:
//
//	space := tune.Space{
//		Hidden:       [][]int{{16}, {32}, {32, 16}},
//		LearningRate: []float64{0.003, 0.01, 0.03},
//		Dropout:      []float64{0, 0.2},
//	}
//	res, err := tune.Search(ctx, data, space.Grid(), tune.Config{Metric: evaluation.MacroF1})
//	best := res.Best()
//	network, scaler, err := tune.Fit(ctx, data, best.Candidate, tune.Config{})
//
// All candidates are evaluated on the same folds. The scaler is fitted on
// the training part of every fold only, so no information leaks from the
// test part. Trials run in parallel; the result does not depend on the
// number of workers.
package tune

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

// Candidate is one hyperparameter configuration.
type Candidate struct {
	Hidden           []int         `json:"hidden"`
	HiddenActivation nn.Activation `json:"hidden_activation"`
	LearningRate     float64       `json:"learning_rate"`
	BatchSize        int           `json:"batch_size"`
	Dropout          float64       `json:"dropout,omitempty"`
	WeightDecay      float64       `json:"weight_decay,omitempty"`
	LayerNorm        bool          `json:"layer_norm,omitempty"`
}

// String describes the candidate in one line.
func (c Candidate) String() string {
	hidden := make([]string, len(c.Hidden))
	for i, h := range c.Hidden {
		hidden[i] = fmt.Sprint(h)
	}
	s := fmt.Sprintf("hidden=[%s] %v lr=%g batch=%d", strings.Join(hidden, " "), c.HiddenActivation, c.LearningRate, c.BatchSize)
	if c.Dropout > 0 {
		s += fmt.Sprintf(" dropout=%g", c.Dropout)
	}
	if c.WeightDecay > 0 {
		s += fmt.Sprintf(" decay=%g", c.WeightDecay)
	}
	if c.LayerNorm {
		s += " layer-norm"
	}
	return s
}

// Space lists the values to try for every hyperparameter. An empty field
// uses one default value: one hidden layer of 32 neurons, tanh, learning
// rate 0.01, batch size 32, no dropout, no weight decay, no layer norm.
type Space struct {
	Hidden           [][]int
	HiddenActivation []nn.Activation
	LearningRate     []float64
	BatchSize        []int
	Dropout          []float64
	WeightDecay      []float64
	LayerNorm        []bool
}

func or[T any](v []T, def T) []T {
	if len(v) == 0 {
		return []T{def}
	}
	return v
}

func (s Space) filled() Space {
	return Space{
		Hidden:           or(s.Hidden, []int{32}),
		HiddenActivation: or(s.HiddenActivation, nn.Tanh),
		LearningRate:     or(s.LearningRate, 0.01),
		BatchSize:        or(s.BatchSize, 32),
		Dropout:          or(s.Dropout, 0),
		WeightDecay:      or(s.WeightDecay, 0),
		LayerNorm:        or(s.LayerNorm, false),
	}
}

// Size returns the number of combinations in the space.
func (s Space) Size() int {
	f := s.filled()
	return len(f.Hidden) * len(f.HiddenActivation) * len(f.LearningRate) * len(f.BatchSize) *
		len(f.Dropout) * len(f.WeightDecay) * len(f.LayerNorm)
}

// candidate returns the combination with the given index, counting through
// the fields like the digits of a number.
func (s Space) candidate(i int) Candidate {
	pick := func(n int) int {
		v := i % n
		i /= n
		return v
	}
	return Candidate{
		Hidden:           slices.Clone(s.Hidden[pick(len(s.Hidden))]),
		HiddenActivation: s.HiddenActivation[pick(len(s.HiddenActivation))],
		LearningRate:     s.LearningRate[pick(len(s.LearningRate))],
		BatchSize:        s.BatchSize[pick(len(s.BatchSize))],
		Dropout:          s.Dropout[pick(len(s.Dropout))],
		WeightDecay:      s.WeightDecay[pick(len(s.WeightDecay))],
		LayerNorm:        s.LayerNorm[pick(len(s.LayerNorm))],
	}
}

// Grid returns every combination of the space.
func (s Space) Grid() []Candidate {
	f := s.filled()
	out := make([]Candidate, f.Size())
	for i := range out {
		out[i] = f.candidate(i)
	}
	return out
}

// Random returns n different combinations drawn uniformly from the space,
// or all of them if the space has at most n. Random search usually finds
// a good configuration with far fewer trials than the full grid.
func (s Space) Random(n int, rng *rand.Rand) []Candidate {
	f := s.filled()
	size := f.Size()
	if n >= size {
		return f.Grid()
	}
	if rng == nil {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	chosen := map[int]bool{}
	out := make([]Candidate, 0, n)
	for len(out) < n {
		i := rng.IntN(size)
		if !chosen[i] {
			chosen[i] = true
			out = append(out, f.candidate(i))
		}
	}
	return out
}

// Config configures Search and Fit. The zero value is usable.
type Config struct {
	// Folds is the number of cross-validation folds. Default 5.
	Folds int
	// Metric compares the candidates. Default: evaluation.Accuracy for
	// classification, evaluation.RMSE for regression.
	Metric evaluation.Metric
	// Epochs is the maximum number of epochs per training. Default 100.
	Epochs int
	// Patience stops a training when the loss on a held-out part of the
	// training data has not improved for this many epochs. Default 10;
	// negative disables early stopping.
	Patience int
	// ValidationRatio is the share of the training data held out for early
	// stopping. Default 0.1.
	ValidationRatio float64
	// Optimizer returns a fresh optimizer per training. Default Adam.
	Optimizer func() nn.Optimizer
	// OutputActivation defaults to Softmax for classification and Linear
	// for regression.
	OutputActivation nn.Activation
	// NoScaling turns off the standardization of the features, which is
	// fitted on the training part of every fold.
	NoScaling bool
	// RegressionThreshold is the error that counts as correct for
	// regression metrics based on correct predictions.
	RegressionThreshold float64
	// Workers is the number of trials trained at the same time. Zero uses
	// GOMAXPROCS. The results do not depend on it.
	Workers int
	// Seed makes the folds and the trainings reproducible. Zero picks a
	// random seed.
	Seed uint64
	// OnTrial is called after every finished trial, from one goroutine at a
	// time but in no particular order.
	OnTrial func(Trial)
}

// Trial is the cross-validated result of one candidate.
type Trial struct {
	Candidate Candidate `json:"candidate"`
	// Scores holds the metric per fold.
	Scores []float64 `json:"scores"`
	// Mean and Std summarize the scores; Std is the sample standard deviation.
	Mean float64 `json:"mean"`
	Std  float64 `json:"std"`
	// Epochs holds the number of epochs trained per fold.
	Epochs   []int         `json:"epochs"`
	Duration time.Duration `json:"duration"`
}

// Result is the outcome of Search.
type Result struct {
	Metric string `json:"metric"`
	Higher bool   `json:"higher"`
	// Trials are in the order of the candidates.
	Trials []Trial `json:"trials"`
	// BestIndex is the index of the best trial.
	BestIndex int `json:"best_index"`
}

// Best returns the trial with the best mean score.
func (r *Result) Best() Trial { return r.Trials[r.BestIndex] }

// Ranked returns the trials from best to worst.
func (r *Result) Ranked() []Trial {
	out := slices.Clone(r.Trials)
	slices.SortStableFunc(out, func(a, b Trial) int {
		switch {
		case better(r.Higher, a.Mean, b.Mean):
			return -1
		case better(r.Higher, b.Mean, a.Mean):
			return 1
		}
		return 0
	})
	return out
}

// better reports whether a beats b; NaN is worse than everything.
func better(higher bool, a, b float64) bool {
	switch {
	case math.IsNaN(a):
		return false
	case math.IsNaN(b):
		return true
	case higher:
		return a > b
	}
	return a < b
}

func (c Config) withDefaults(task nn.Task) Config {
	if c.Folds == 0 {
		c.Folds = 5
	}
	if c.Metric.Value == nil {
		c.Metric = evaluation.Accuracy
		if task == nn.Regression {
			c.Metric = evaluation.RMSE
		}
	}
	if c.Epochs <= 0 {
		c.Epochs = 100
	}
	if c.Patience == 0 {
		c.Patience = 10
	}
	if c.ValidationRatio == 0 {
		c.ValidationRatio = 0.1
	}
	if c.Optimizer == nil {
		c.Optimizer = nn.DefaultAdam
	}
	if c.OutputActivation == nn.Sigmoid {
		c.OutputActivation = nn.Softmax
		if task == nn.Regression {
			c.OutputActivation = nn.Linear
		}
	}
	if c.Seed == 0 {
		c.Seed = rand.Uint64()
	}
	return c
}

// fold holds the prepared data of one cross-validation fold.
type fold struct {
	train, val, test *learn.Set
}

// prepare splits off the validation part and scales the data with a scaler
// fitted on the training part.
func prepare(train, test *learn.Set, cfg Config, rng *rand.Rand) (fold, *learn.Scaler, error) {
	f := fold{train: train, test: test}
	var sc *learn.Scaler
	if !cfg.NoScaling {
		var err error
		if sc, err = learn.FitScaler(train, learn.Standardize); err != nil {
			return f, nil, err
		}
		if f.train, err = sc.TransformSet(train); err != nil {
			return f, nil, err
		}
		if test != nil {
			if f.test, err = sc.TransformSet(test); err != nil {
				return f, nil, err
			}
		}
	}
	if cfg.Patience > 0 && f.train.Len() >= 4 {
		var err error
		if f.train, f.val, err = f.train.StratifiedSplit(1-cfg.ValidationRatio, rng); err != nil {
			return f, nil, err
		}
	}
	return f, sc, nil
}

// train fits a network for a candidate on prepared data.
func train(ctx context.Context, f fold, task nn.Task, labels []string, c Candidate, cfg Config, seed uint64, workers int) (*nn.Network, *nn.History, error) {
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	netCfg := nn.Config{
		Inputs: f.train.InputSize(), Hidden: c.Hidden, HiddenActivation: c.HiddenActivation,
		OutputActivation: cfg.OutputActivation, Dropout: c.Dropout, LayerNorm: c.LayerNorm, Rand: rng,
	}
	if task == nn.Classification {
		netCfg.Labels = labels
	} else {
		netCfg.Outputs = 1
	}
	n, err := nn.New(netCfg)
	if err != nil {
		return nil, nil, err
	}
	fit := nn.FitConfig{
		Epochs: cfg.Epochs, BatchSize: c.BatchSize, Optimizer: cfg.Optimizer(), LearningRate: c.LearningRate,
		WeightDecay: c.WeightDecay, Rand: rng, Workers: workers,
	}
	var val nn.Data
	if f.val != nil {
		val, fit.Patience = f.val.Data(), max(cfg.Patience, 0)
	}
	h, err := nn.Fit(ctx, n, f.train.Data(), val, fit)
	return n, h, err
}

// Search evaluates every candidate with k-fold cross-validation on the set
// and returns the results. It can be cancelled through ctx.
func Search(ctx context.Context, set *learn.Set, candidates []Candidate, cfg Config) (*Result, error) {
	if set == nil || set.Len() == 0 {
		return nil, errors.New("tune: no data")
	}
	if len(candidates) == 0 {
		return nil, errors.New("tune: no candidates")
	}
	cfg = cfg.withDefaults(set.Task)
	if cfg.Folds < 2 {
		return nil, fmt.Errorf("tune: need at least 2 folds, got %d", cfg.Folds)
	}
	rng := rand.New(rand.NewPCG(cfg.Seed, cfg.Seed+1))
	splits, err := set.KFold(cfg.Folds, rng)
	if err != nil {
		return nil, fmt.Errorf("tune: %w", err)
	}
	folds := make([]fold, len(splits))
	for i, s := range splits {
		if folds[i], _, err = prepare(s.Train, s.Test, cfg, rng); err != nil {
			return nil, fmt.Errorf("tune: fold %d: %w", i, err)
		}
	}

	workers := cfg.Workers
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	workers = min(workers, len(candidates))
	fitWorkers := 0 // a single trial may use all cores
	if workers > 1 {
		fitWorkers = 1
	}

	res := &Result{Metric: cfg.Metric.Name, Higher: cfg.Metric.Higher, Trials: make([]Trial, len(candidates))}
	var (
		mu       sync.Mutex
		firstErr error
		next     int
		wg       sync.WaitGroup
	)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				mu.Lock()
				i := next
				next++
				stop := firstErr != nil
				mu.Unlock()
				if stop || i >= len(candidates) || ctx.Err() != nil {
					return
				}
				t, err := evaluate(ctx, folds, set, candidates[i], cfg, uint64(i), fitWorkers)
				mu.Lock()
				if err != nil && firstErr == nil {
					firstErr = fmt.Errorf("tune: candidate %d (%v): %w", i, candidates[i], err)
				}
				res.Trials[i] = t
				if err == nil && cfg.OnTrial != nil {
					cfg.OnTrial(t)
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if firstErr != nil {
		return nil, firstErr
	}
	for i, t := range res.Trials {
		if better(res.Higher, t.Mean, res.Trials[res.BestIndex].Mean) {
			res.BestIndex = i
		}
	}
	return res, nil
}

// evaluate cross-validates one candidate.
func evaluate(ctx context.Context, folds []fold, set *learn.Set, c Candidate, cfg Config, index uint64, workers int) (Trial, error) {
	start := time.Now()
	t := Trial{Candidate: c, Scores: make([]float64, len(folds)), Epochs: make([]int, len(folds))}
	for k, f := range folds {
		n, h, err := train(ctx, f, set.Task, set.Labels, c, cfg, cfg.Seed^(index<<20)^uint64(k+1)*0x9e3779b97f4a7c15, workers)
		if err != nil {
			return t, err
		}
		e, err := evaluation.Evaluate(n, f.test, cfg.RegressionThreshold)
		if err != nil {
			return t, err
		}
		t.Scores[k], t.Epochs[k] = cfg.Metric.Value(e), len(h.Epochs)
	}
	for _, s := range t.Scores {
		t.Mean += s
	}
	t.Mean /= float64(len(t.Scores))
	for _, s := range t.Scores {
		t.Std += (s - t.Mean) * (s - t.Mean)
	}
	t.Std = math.Sqrt(t.Std / float64(len(t.Scores)-1))
	t.Duration = time.Since(start)
	return t, nil
}

// Fit trains a network for a candidate on the whole set, e.g. the best
// candidate of a Search. It returns the network and the scaler that must be
// applied to its inputs (nil with Config.NoScaling).
func Fit(ctx context.Context, set *learn.Set, c Candidate, cfg Config) (*nn.Network, *learn.Scaler, error) {
	if set == nil || set.Len() == 0 {
		return nil, nil, errors.New("tune: no data")
	}
	cfg = cfg.withDefaults(set.Task)
	rng := rand.New(rand.NewPCG(cfg.Seed, cfg.Seed+2))
	f, sc, err := prepare(set, nil, cfg, rng)
	if err != nil {
		return nil, nil, fmt.Errorf("tune: %w", err)
	}
	n, _, err := train(ctx, f, set.Task, set.Labels, c, cfg, cfg.Seed+3, cfg.Workers)
	if err != nil {
		return nil, nil, err
	}
	return n, sc, nil
}
