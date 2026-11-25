package explain

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

// BenchmarkPermutation measures one importance computation with 5 repeats
// for networks of different widths on 1,000 test samples.
func BenchmarkPermutation(b *testing.B) {
	for _, features := range []int{10, 50, 200} {
		b.Run(fmt.Sprintf("features=%d", features), func(b *testing.B) {
			rng := rand.New(rand.NewPCG(1, 1))
			s := learn.NewSet(nn.Regression)
			for range 1000 {
				x := make([]float64, features)
				for j := range x {
					x[j] = rng.NormFloat64()
				}
				_ = s.Add(learn.NewRegressionSample(x, x[0]))
			}
			n, _ := nn.New(nn.Config{Inputs: features, Hidden: []int{32}, Outputs: 1, OutputActivation: nn.Linear, Rand: rng})
			for b.Loop() {
				_, _ = Permutation(n, s, Config{Repeats: 5, Rand: rng})
			}
		})
	}
}
