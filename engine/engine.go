// Package engine trains networks on a fixed data set and keeps the best one.
//
// Training starts with a high learning rate which decays linearly. After each
// round the network is evaluated on a held-out validation split and the best
// network according to a Criterion is kept.
package engine

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/internal/train"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

// Criterion decides which of two evaluations is better.
type Criterion int

const (
	// Accuracy compares the mean per-class accuracy. Not suited for unbalanced data.
	Accuracy Criterion = iota
	// BalancedAccuracy compares the mean balanced accuracy.
	BalancedAccuracy
	// FMeasure compares the macro-averaged F1 score.
	FMeasure
	// Simple compares the ratio of correct predictions.
	Simple
	// Distance compares the summed distance to the ideal output (lower is better).
	Distance
)

var criterionNames = [...]string{"accuracy", "balanced-accuracy", "f-measure", "simple", "distance"}

// String returns the name of the criterion.
func (c Criterion) String() string {
	if c < 0 || int(c) >= len(criterionNames) {
		return fmt.Sprintf("Criterion(%d)", int(c))
	}
	return criterionNames[c]
}

// better reports whether candidate beats current.
func (c Criterion) better(current, candidate *evaluation.Evaluation) bool {
	if current == nil {
		return true
	}
	switch c {
	case Accuracy:
		return candidate.OverallAccuracy() > current.OverallAccuracy()
	case BalancedAccuracy:
		return candidate.OverallBalancedAccuracy() > current.OverallBalancedAccuracy()
	case FMeasure:
		return candidate.OverallFMeasure() > current.OverallFMeasure()
	case Simple:
		return candidate.CorrectRatio() > current.CorrectRatio()
	case Distance:
		return candidate.TotalDistance < current.TotalDistance
	}
	return false
}

// Config controls the training process.
type Config struct {
	// Tries is the number of independently initialized networks to train.
	Tries int `json:"tries"`
	// Epochs is the number of passes over the training data per round.
	Epochs int `json:"epochs"`
	// TrainingSplit is the ratio of samples used for training; the rest validates.
	TrainingSplit float64 `json:"training_split"`
	// LearningRate is the initial learning rate.
	LearningRate float64 `json:"learning_rate"`
	// Decay is subtracted from the learning rate after each round until it reaches zero.
	Decay float64 `json:"decay"`
	// RegressionThreshold is the maximum error for a regression prediction to count as correct.
	RegressionThreshold float64 `json:"regression_threshold"`
	// BatchSize is the mini-batch size. Values <= 1 mean stochastic gradient descent.
	BatchSize int `json:"batch_size"`
	// HiddenActivation and OutputActivation configure the network. Both default to Sigmoid.
	HiddenActivation nn.Activation `json:"hidden_activation"`
	OutputActivation nn.Activation `json:"output_activation"`
	// Seed makes training reproducible. Zero uses a random seed.
	Seed uint64 `json:"seed"`
	// Logger receives progress information. Nil disables logging.
	Logger *slog.Logger `json:"-"`
}

// DefaultConfig returns the default configuration.
func DefaultConfig() Config {
	return Config{
		Tries:               1,
		Epochs:              100,
		TrainingSplit:       0.7,
		LearningRate:        0.4,
		Decay:               0.005,
		RegressionThreshold: 0.05,
	}
}

// Validate checks the configuration.
func (c Config) Validate() error {
	switch {
	case c.Tries < 1:
		return errors.New("engine: Tries must be at least 1")
	case c.Epochs < 1:
		return errors.New("engine: Epochs must be at least 1")
	case c.TrainingSplit <= 0 || c.TrainingSplit >= 1:
		return errors.New("engine: TrainingSplit must be in (0, 1)")
	case c.LearningRate <= 0:
		return errors.New("engine: LearningRate must be positive")
	case c.Decay <= 0:
		return errors.New("engine: Decay must be positive")
	}
	return nil
}

// Engine trains networks and keeps the best one.
type Engine struct {
	cfg    Config
	data   *learn.Set
	hidden []int

	winner     *nn.Network
	winnerEval *evaluation.Evaluation
}

// New creates an engine for the data set. hidden holds the sizes of the hidden layers.
func New(data *learn.Set, hidden []int, cfg Config) (*Engine, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if data == nil || data.Len() < 2 {
		return nil, errors.New("engine: data set needs at least two samples")
	}
	if data.OutputSize() == 0 {
		return nil, errors.New("engine: data set has no classes")
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.New(slog.DiscardHandler)
	}
	return &Engine{cfg: cfg, data: data, hidden: append([]int(nil), hidden...)}, nil
}

// Config returns the configuration of the engine.
func (e *Engine) Config() Config { return e.cfg }

// Winner returns the best network and its validation result.
// Both are nil before Run completed at least one round.
func (e *Engine) Winner() (*nn.Network, *evaluation.Evaluation) {
	return e.winner, e.winnerEval
}

// Run trains the networks. It can be cancelled through ctx; the best
// network found so far stays available through Winner.
func (e *Engine) Run(ctx context.Context, criterion Criterion) error {
	rng := train.NewRand(e.cfg.Seed)
	training, validation, err := e.data.StratifiedSplit(e.cfg.TrainingSplit, rng)
	if err != nil {
		return err
	}
	log := e.cfg.Logger
	for try := range e.cfg.Tries {
		network, err := nn.New(nn.Config{
			Inputs:           e.data.InputSize(),
			Hidden:           e.hidden,
			Outputs:          e.data.OutputSize(),
			Labels:           e.data.Labels,
			HiddenActivation: e.cfg.HiddenActivation,
			OutputActivation: e.cfg.OutputActivation,
			Rand:             rng,
		})
		if err != nil {
			return err
		}
		log.Info("start try", "try", try+1, "training", training.Len(), "validation", validation.Len())
		for lr := e.cfg.LearningRate; lr > 0; lr -= e.cfg.Decay {
			var loss float64
			for range e.cfg.Epochs {
				if loss, err = train.Epoch(ctx, network, training, lr, e.cfg.BatchSize, rng); err != nil {
					return err
				}
			}
			eval, err := evaluation.Evaluate(network, validation, e.cfg.RegressionThreshold)
			if err != nil {
				return err
			}
			log.Debug("round", "try", try+1, "learning_rate", lr, "loss", loss)
			if criterion.better(e.winnerEval, eval) {
				e.winner, e.winnerEval = network.Clone(), eval
				log.Info("new best network",
					"try", try+1,
					"learning_rate", lr,
					"accuracy", eval.OverallAccuracy(),
					"balanced_accuracy", eval.OverallBalancedAccuracy(),
					"f1", eval.OverallFMeasure(),
					"correct_ratio", eval.CorrectRatio(),
					"distance", eval.TotalDistance)
			}
		}
	}
	return nil
}
