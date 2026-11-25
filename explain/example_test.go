package explain_test

import (
	"context"
	"fmt"
	"math/rand/v2"

	"github.com/breskos/gopher-learn/explain"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

// The price of a flat depends on its area and floor; the number on the
// door does not matter.
func Example() {
	rng := rand.New(rand.NewPCG(1, 1))
	set := func(n int) *learn.Set {
		s := learn.NewSet(nn.Regression)
		s.Features = []string{"area", "floor", "door"}
		for range n {
			area, floor, door := rng.Float64(), rng.Float64(), rng.Float64()
			_ = s.Add(learn.NewRegressionSample([]float64{area, floor, door}, 0.8*area+0.2*floor))
		}
		return s
	}
	train, test := set(2000), set(500)
	network, _ := nn.New(nn.Config{Inputs: 3, Hidden: []int{8}, Outputs: 1, HiddenActivation: nn.Tanh, OutputActivation: nn.Linear, Rand: rng})
	if _, err := nn.Fit(context.Background(), network, train.Data(), nn.Data{}, nn.FitConfig{Epochs: 50, Rand: rng}); err != nil {
		panic(err)
	}

	imp, err := explain.Permutation(network, test, explain.Config{Rand: rng})
	if err != nil {
		panic(err)
	}
	for _, f := range imp {
		fmt.Printf("%s matters: %v\n", f.Feature, f.Mean > 0.01)
	}
	// Output:
	// area matters: true
	// floor matters: true
	// door matters: false
}
