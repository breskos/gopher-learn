// Package online trains a network continuously from a stream of samples.
//
// Learn handles one sample of a stream with the test-then-train protocol: it
// predicts the sample, records the result in a prequential evaluation and a
// drift detector, then stores the sample and applies a few quick training
// steps. Iterate retrains on the stored samples. The memory of stored
// samples is bounded by a sliding window or a reservoir, and a detected
// concept drift makes the learner forget the samples from before the change.
//
// Inject stores and trains on a sample without evaluating it first.
package online

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math/rand/v2"

	"github.com/breskos/gopher-learn/drift"
	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/internal/train"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

var (
	// ErrSampleExists is returned by Inject for a duplicate sample without force.
	ErrSampleExists = errors.New("online: sample already exists")
	// ErrNotEnoughData is returned by Iterate if the set has fewer than MinimumDataPoints samples.
	ErrNotEnoughData = errors.New("online: not enough data points")
)

// Config controls the online learner.
type Config struct {
	// HotShots is the number of training steps applied to a newly injected sample.
	HotShots int `json:"hot_shots"`
	// HotShotRate is the learning rate used for those steps.
	HotShotRate float64 `json:"hot_shot_rate"`
	// TrainingSplit is the ratio of samples used for training in Iterate.
	TrainingSplit float64 `json:"training_split"`
	// MinimumDataPoints is the number of samples required before Iterate trains.
	MinimumDataPoints int `json:"minimum_data_points"`
	// MinEpochs and MaxEpochs bound the random number of epochs per iteration.
	MinEpochs int `json:"min_epochs"`
	MaxEpochs int `json:"max_epochs"`
	// MinLearningRate and MaxLearningRate bound the random learning rate per iteration.
	MinLearningRate float64 `json:"min_learning_rate"`
	MaxLearningRate float64 `json:"max_learning_rate"`
	// TargetScore is the score Init trains towards.
	TargetScore float64 `json:"target_score"`
	// MaxInitLoops bounds the number of iterations of Init.
	MaxInitLoops int `json:"max_init_loops"`
	// RegressionThreshold is the maximum error for a regression prediction to count as correct.
	RegressionThreshold float64 `json:"regression_threshold"`
	// BatchSize is the mini-batch size. Values <= 1 mean stochastic gradient descent.
	BatchSize int `json:"batch_size"`
	// HiddenActivation and OutputActivation configure the network. Both default to Sigmoid.
	HiddenActivation nn.Activation `json:"hidden_activation"`
	OutputActivation nn.Activation `json:"output_activation"`
	// Memory bounds the number of stored samples. Default: a sliding window.
	Memory Memory `json:"memory"`
	// MemorySize is the maximum number of stored samples for Window and
	// Reservoir memory. Default 5000.
	MemorySize int `json:"memory_size"`
	// Drift selects the drift detector fed with the prequential errors of
	// Learn. Default ADWIN.
	Drift Detector `json:"drift"`
	// DriftDelta is the confidence of ADWIN (default 0.002) or the
	// threshold of Page-Hinkley (default 50); zero uses the default.
	DriftDelta float64 `json:"drift_delta,omitempty"`
	// PrequentialWindow is the number of recent results for the windowed
	// prequential metrics. Default 500.
	PrequentialWindow int `json:"prequential_window"`
	// DriftCooldown is the number of samples after a detected drift during
	// which further detections are ignored, so one change is reported once.
	// Default 200.
	DriftCooldown int `json:"drift_cooldown"`
	// OnDrift is called after Learn detected a drift and the memory reacted.
	OnDrift func(DriftEvent) `json:"-"`
	// Seed makes training reproducible. Zero uses a random seed.
	Seed uint64 `json:"seed"`
	// Logger receives progress information. Nil disables logging.
	Logger *slog.Logger `json:"-"`
}

// DefaultConfig returns the default configuration.
func DefaultConfig() Config {
	return Config{
		HotShots:            5,
		HotShotRate:         0.5,
		TrainingSplit:       0.7,
		MinimumDataPoints:   10,
		MinEpochs:           10,
		MaxEpochs:           30,
		MinLearningRate:     0.2,
		MaxLearningRate:     0.5,
		TargetScore:         0.7,
		MaxInitLoops:        5,
		RegressionThreshold: 0.05,
		Memory:              Window,
		MemorySize:          5000,
		Drift:               ADWIN,
		PrequentialWindow:   500,
		DriftCooldown:       200,
	}
}

// Validate checks the configuration.
func (c Config) Validate() error {
	switch {
	case c.TrainingSplit <= 0 || c.TrainingSplit >= 1:
		return errors.New("online: TrainingSplit must be in (0, 1)")
	case c.MinEpochs < 1 || c.MaxEpochs < c.MinEpochs:
		return errors.New("online: need 1 <= MinEpochs <= MaxEpochs")
	case c.MinLearningRate <= 0 || c.MaxLearningRate < c.MinLearningRate:
		return errors.New("online: need 0 < MinLearningRate <= MaxLearningRate")
	case c.MinimumDataPoints < 2:
		return errors.New("online: MinimumDataPoints must be at least 2")
	case c.Memory < Unbounded || c.Memory > Reservoir:
		return fmt.Errorf("online: invalid memory %v", c.Memory)
	case c.Memory != Unbounded && c.MemorySize < c.MinimumDataPoints:
		return errors.New("online: MemorySize must be at least MinimumDataPoints")
	case c.Drift < NoDrift || c.Drift > PageHinkley:
		return fmt.Errorf("online: invalid drift detector %v", c.Drift)
	case c.DriftDelta < 0:
		return errors.New("online: DriftDelta must not be negative")
	case c.DriftCooldown < 0:
		return errors.New("online: DriftCooldown must not be negative")
	case c.PrequentialWindow < 1:
		return errors.New("online: PrequentialWindow must be at least 1")
	}
	return nil
}

// Online is a continuously learning network.
type Online struct {
	Config         Config                 `json:"config"`
	Hidden         []int                  `json:"hidden"`
	Data           *learn.Set             `json:"data"`
	Network        *nn.Network            `json:"network"`
	LastEvaluation *evaluation.Evaluation `json:"last_evaluation,omitempty"`
	// Prequential holds the test-then-train results of Learn.
	Prequential *evaluation.Prequential `json:"prequential,omitempty"`
	// DriftADWIN or DriftPageHinkley holds the state of the drift detector,
	// depending on Config.Drift.
	DriftADWIN       *drift.ADWIN       `json:"drift_adwin,omitempty"`
	DriftPageHinkley *drift.PageHinkley `json:"drift_page_hinkley,omitempty"`
	// Seen counts the samples offered to the memory since it was last
	// cleared; reservoir sampling needs it.
	Seen int `json:"seen"`
	// Learned counts the samples processed by Learn.
	Learned int `json:"learned"`
	// Drifts counts the detected drifts; LastDrift is the value of Learned
	// at the last one.
	Drifts    int `json:"drifts"`
	LastDrift int `json:"last_drift"`

	rng *rand.Rand
}

// New creates an online learner. The data set defines the task and, for
// classification, the classes; it may be empty. inputs is the size of the
// feature vectors.
func New(data *learn.Set, inputs int, hidden []int, cfg Config) (*Online, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if data == nil {
		return nil, errors.New("online: nil data set")
	}
	o := &Online{Config: cfg, Hidden: append([]int(nil), hidden...), Data: data}
	net, err := nn.New(nn.Config{
		Inputs:           inputs,
		Hidden:           hidden,
		Outputs:          data.OutputSize(),
		Labels:           data.Labels,
		HiddenActivation: cfg.HiddenActivation,
		OutputActivation: cfg.OutputActivation,
		Rand:             o.random(),
	})
	if err != nil {
		return nil, err
	}
	o.Network = net
	return o, nil
}

func (o *Online) random() *rand.Rand {
	if o.rng == nil {
		o.rng = train.NewRand(o.Config.Seed)
	}
	return o.rng
}

func (o *Online) logger() *slog.Logger {
	if o.Config.Logger == nil {
		return slog.New(slog.DiscardHandler)
	}
	return o.Config.Logger
}

// Init iterates until the score reaches Config.TargetScore or
// Config.MaxInitLoops is exhausted and returns the last score.
func (o *Online) Init(ctx context.Context) (float64, error) {
	var score float64
	for range max(o.Config.MaxInitLoops, 1) {
		var err error
		if score, err = o.Iterate(ctx); err != nil {
			return score, err
		}
		if score >= o.Config.TargetScore {
			break
		}
	}
	return score, nil
}

// Inject adds a sample to the data set and trains the network on it.
// A sample whose feature vector already exists is rejected unless force is set.
func (o *Online) Inject(sample *learn.Sample, force bool) error {
	if sample == nil {
		return errors.New("online: nil sample")
	}
	// Check before touching the data set so a bad sample cannot corrupt it.
	if len(sample.Vector) != o.Network.InputSize() {
		return fmt.Errorf("online: %w: sample has %d features, network expects %d", nn.ErrDimension, len(sample.Vector), o.Network.InputSize())
	}
	if !force && o.Data.Contains(sample) {
		return ErrSampleExists
	}
	if err := o.Data.Prepare(sample); err != nil {
		return fmt.Errorf("online: %w", err)
	}
	if err := o.remember(sample); err != nil {
		return err
	}
	return o.hotShots(sample)
}

func (o *Online) hotShots(sample *learn.Sample) error {
	for range o.Config.HotShots {
		if _, err := o.Network.Train(sample.Vector, sample.Output, o.Config.HotShotRate); err != nil {
			return fmt.Errorf("online: %w", err)
		}
	}
	return nil
}

// Iterate retrains the network on a random split of the collected data and
// returns its score on the remaining samples: the macro F1 score for
// classification, the ratio of correct predictions for regression.
func (o *Online) Iterate(ctx context.Context) (float64, error) {
	if o.Data.Len() < o.Config.MinimumDataPoints {
		return 0, ErrNotEnoughData
	}
	rng := o.random()
	training, testing, err := o.Data.StratifiedSplit(o.Config.TrainingSplit, rng)
	if err != nil {
		return 0, err
	}
	lr := o.Config.MinLearningRate + rng.Float64()*(o.Config.MaxLearningRate-o.Config.MinLearningRate)
	epochs := o.Config.MinEpochs + rng.IntN(o.Config.MaxEpochs-o.Config.MinEpochs+1)
	for range epochs {
		if _, err := train.Epoch(ctx, o.Network, training, lr, o.Config.BatchSize, rng); err != nil {
			return 0, err
		}
	}
	eval, err := evaluation.Evaluate(o.Network, testing, o.Config.RegressionThreshold)
	if err != nil {
		return 0, err
	}
	o.LastEvaluation = eval
	score := eval.CorrectRatio()
	if o.Data.Task == nn.Classification {
		score = eval.OverallFMeasure()
	}
	o.logger().Info("iteration", "samples", o.Data.Len(), "epochs", epochs, "learning_rate", lr, "score", score)
	return score, nil
}
