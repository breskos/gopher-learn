// Package train contains the training loop shared by engine and online.
package train

import (
	"context"
	"math/rand/v2"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

// NewRand returns a generator for the given seed, or a randomly seeded one for seed 0.
func NewRand(seed uint64) *rand.Rand {
	if seed == 0 {
		return rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	return rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
}

// Epoch trains the network once on every sample of the set in random order
// and returns the mean loss. batchSize <= 1 means plain SGD.
func Epoch(ctx context.Context, n *nn.Network, set *learn.Set, lr float64, batchSize int, rng *rand.Rand) (float64, error) {
	batchSize = max(batchSize, 1)
	order := rng.Perm(set.Len())
	xs := make([][]float64, 0, batchSize)
	ys := make([][]float64, 0, batchSize)
	var total float64
	var batches int
	for i, idx := range order {
		s := set.Samples[idx]
		xs = append(xs, s.Vector)
		ys = append(ys, s.Output)
		if len(xs) < batchSize && i < len(order)-1 {
			continue
		}
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		loss, err := n.TrainBatch(xs, ys, lr)
		if err != nil {
			return 0, err
		}
		total += loss
		batches++
		xs, ys = xs[:0], ys[:0]
	}
	if batches == 0 {
		return 0, nil
	}
	return total / float64(batches), nil
}
