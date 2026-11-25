package learn

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/breskos/gopher-learn/nn"
)

// Fold is one train/test partition of a cross-validation.
type Fold struct {
	Train *Set
	Test  *Set
}

func randOrNew(rng *rand.Rand) *rand.Rand {
	if rng == nil {
		return rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	return rng
}

// groups returns the shuffled sample indices per class. Regression sets form
// a single group.
func (s *Set) groups(rng *rand.Rand) [][]int {
	if s.Task != nn.Classification {
		return [][]int{rng.Perm(s.Len())}
	}
	g := make([][]int, len(s.Labels))
	for _, i := range rng.Perm(s.Len()) {
		c := s.Samples[i].Class
		g[c] = append(g[c], i)
	}
	return g
}

// StratifiedSplit divides a classification set like Split, but keeps the
// class proportions in both parts. Every class with at least two samples
// appears in both parts. Regression sets are split like Split.
func (s *Set) StratifiedSplit(ratio float64, rng *rand.Rand) (*Set, *Set, error) {
	if s.Task != nn.Classification {
		return s.Split(ratio, rng)
	}
	if ratio <= 0 || ratio >= 1 {
		return nil, nil, fmt.Errorf("learn: split ratio must be in (0, 1), got %v", ratio)
	}
	rng = randOrNew(rng)
	var first, second []int
	for _, g := range s.groups(rng) {
		cut := int(math.Round(ratio * float64(len(g))))
		if len(g) >= 2 {
			cut = min(max(cut, 1), len(g)-1)
		}
		first = append(first, g[:cut]...)
		second = append(second, g[cut:]...)
	}
	// Mix the classes so training does not see them in blocks.
	rng.Shuffle(len(first), func(i, j int) { first[i], first[j] = first[j], first[i] })
	rng.Shuffle(len(second), func(i, j int) { second[i], second[j] = second[j], second[i] })
	return s.subset(first), s.subset(second), nil
}

// KFold divides the set into k folds for cross-validation. Each fold uses
// one k-th of the samples as test set and the rest for training. For
// classification the folds are stratified: the classes are dealt evenly
// across the folds.
func (s *Set) KFold(k int, rng *rand.Rand) ([]Fold, error) {
	if k < 2 {
		return nil, errors.New("learn: k must be at least 2")
	}
	if k > s.Len() {
		return nil, fmt.Errorf("learn: k = %d exceeds the number of samples (%d)", k, s.Len())
	}
	rng = randOrNew(rng)
	parts := make([][]int, k)
	next := 0
	for _, g := range s.groups(rng) {
		for _, i := range g {
			parts[next] = append(parts[next], i)
			next = (next + 1) % k
		}
	}
	folds := make([]Fold, k)
	for f := range k {
		var train []int
		for p, part := range parts {
			if p != f {
				train = append(train, part...)
			}
		}
		rng.Shuffle(len(train), func(i, j int) { train[i], train[j] = train[j], train[i] })
		folds[f] = Fold{Train: s.subset(train), Test: s.subset(parts[f])}
	}
	return folds, nil
}
