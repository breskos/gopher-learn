package nn

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"time"
)

// Data holds inputs and target vectors for Fit.
type Data struct {
	X [][]float64
	Y [][]float64
}

// Len returns the number of samples.
func (d Data) Len() int { return len(d.X) }

// Schedule returns the learning rate for an epoch (starting at 0) from the
// base learning rate.
type Schedule func(epoch int, base float64) float64

// StepDecay multiplies the learning rate by factor every `every` epochs.
func StepDecay(every int, factor float64) Schedule {
	return func(epoch int, base float64) float64 {
		return base * math.Pow(factor, float64(epoch/max(every, 1)))
	}
}

// CosineDecay lowers the learning rate along a cosine curve from base to
// zero over the given number of epochs.
func CosineDecay(epochs int) Schedule {
	return func(epoch int, base float64) float64 {
		return base * 0.5 * (1 + math.Cos(math.Pi*float64(min(epoch, epochs))/float64(max(epochs, 1))))
	}
}

// FitConfig configures Fit. The zero value is usable: Adam with learning
// rate 0.01, mini-batches of 32 and at most 100 epochs.
type FitConfig struct {
	// Epochs is the maximum number of passes over the training data. Default 100.
	Epochs int
	// BatchSize is the mini-batch size. Default 32.
	BatchSize int
	// Optimizer defaults to DefaultAdam.
	Optimizer Optimizer
	// LearningRate is the base learning rate. Default 0.01.
	LearningRate float64
	// Schedule adjusts the learning rate per epoch. Nil keeps it constant.
	Schedule Schedule
	// WeightDecay is decoupled L2 regularization, e.g. 1e-4.
	WeightDecay float64
	// Patience stops training when the validation loss has not improved for
	// this many epochs and restores the best weights. Zero disables early
	// stopping. It needs validation data.
	Patience int
	// MinDelta is the smallest decrease of the validation loss that counts
	// as an improvement.
	MinDelta float64
	// Rand shuffles the training data. Nil uses a randomly seeded source.
	Rand *rand.Rand
	// OnEpoch is called after every epoch. Returning false stops training.
	OnEpoch func(EpochStats) bool
}

// EpochStats describes one finished epoch.
type EpochStats struct {
	Epoch        int           `json:"epoch"`
	LearningRate float64       `json:"learning_rate"`
	TrainLoss    float64       `json:"train_loss"`
	ValLoss      float64       `json:"val_loss"`
	HasVal       bool          `json:"has_val"`
	Duration     time.Duration `json:"duration"`
}

// StopReason tells why Fit ended.
type StopReason string

const (
	StopEpochs   StopReason = "max epochs reached"
	StopPatience StopReason = "validation loss stopped improving"
	StopCallback StopReason = "stopped by OnEpoch"
	StopDiverged StopReason = "loss diverged (NaN or infinite)"
)

// History records the course of a training run.
type History struct {
	Epochs []EpochStats `json:"epochs"`
	// BestEpoch is the epoch with the lowest validation loss (or training
	// loss without validation data); its weights are in the network.
	BestEpoch int        `json:"best_epoch"`
	Stop      StopReason `json:"stop"`
}

// Fit trains the network on train and returns the training history. If val
// has samples, the validation loss is computed after every epoch and the
// weights of the best epoch are kept. Fit can be cancelled through ctx.
func Fit(ctx context.Context, n *Network, train, val Data, cfg FitConfig) (*History, error) {
	if train.Len() == 0 {
		return nil, errors.New("nn: no training data")
	}
	if len(train.X) != len(train.Y) || len(val.X) != len(val.Y) {
		return nil, errors.New("nn: inputs and targets differ in length")
	}
	if cfg.Patience > 0 && val.Len() == 0 {
		return nil, errors.New("nn: early stopping needs validation data")
	}
	if cfg.Epochs <= 0 {
		cfg.Epochs = 100
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 32
	}
	if cfg.LearningRate <= 0 {
		cfg.LearningRate = 0.01
	}
	if cfg.Optimizer == nil {
		cfg.Optimizer = DefaultAdam()
	}
	rng := cfg.Rand
	if rng == nil {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}

	h := &History{Stop: StopEpochs}
	best := math.Inf(1)
	var bestWeights *Network
	sinceBest := 0
	xs := make([][]float64, 0, cfg.BatchSize)
	ys := make([][]float64, 0, cfg.BatchSize)
	for epoch := range cfg.Epochs {
		start := time.Now()
		lr := cfg.LearningRate
		if cfg.Schedule != nil {
			lr = cfg.Schedule(epoch, cfg.LearningRate)
		}
		step := Step{Optimizer: cfg.Optimizer, LearningRate: lr, WeightDecay: cfg.WeightDecay}
		var total float64
		order := rng.Perm(train.Len())
		for i, idx := range order {
			xs, ys = append(xs, train.X[idx]), append(ys, train.Y[idx])
			if len(xs) < cfg.BatchSize && i < len(order)-1 {
				continue
			}
			if err := ctx.Err(); err != nil {
				return h, err
			}
			loss, err := n.TrainBatchWith(xs, ys, step)
			if err != nil {
				return h, err
			}
			total += loss * float64(len(xs))
			xs, ys = xs[:0], ys[:0]
		}
		stats := EpochStats{Epoch: epoch, LearningRate: lr, TrainLoss: total / float64(train.Len())}
		if val.Len() > 0 {
			vl, err := n.Loss(val.X, val.Y)
			if err != nil {
				return h, fmt.Errorf("nn: validation: %w", err)
			}
			stats.ValLoss, stats.HasVal = vl, true
		}
		stats.Duration = time.Since(start)
		h.Epochs = append(h.Epochs, stats)

		monitored := stats.TrainLoss
		if stats.HasVal {
			monitored = stats.ValLoss
		}
		if math.IsNaN(monitored) || math.IsInf(monitored, 0) {
			h.Stop = StopDiverged
			break
		}
		if monitored < best-cfg.MinDelta {
			best, h.BestEpoch, sinceBest = monitored, epoch, 0
			if stats.HasVal {
				bestWeights = n.Clone()
			}
		} else {
			sinceBest++
		}
		if cfg.OnEpoch != nil && !cfg.OnEpoch(stats) {
			h.Stop = StopCallback
			break
		}
		if cfg.Patience > 0 && sinceBest >= cfg.Patience {
			h.Stop = StopPatience
			break
		}
	}
	if bestWeights != nil {
		for i, l := range bestWeights.Layers {
			copy(n.Layers[i].Weights, l.Weights)
			copy(n.Layers[i].Biases, l.Biases)
		}
	}
	return h, nil
}
