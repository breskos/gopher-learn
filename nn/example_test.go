package nn_test

import (
	"fmt"
	"math/rand/v2"

	"github.com/breskos/gopher-learn/nn"
)

func Example() {
	net, err := nn.New(nn.Config{
		Inputs:           2,
		Hidden:           []int{8},
		Labels:           []string{"same", "different"},
		HiddenActivation: nn.Tanh,
		OutputActivation: nn.Softmax,
		Rand:             rand.New(rand.NewPCG(1, 2)),
	})
	if err != nil {
		panic(err)
	}

	xs := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	ys := [][]float64{{1, 0}, {0, 1}, {0, 1}, {1, 0}}
	for range 5000 {
		if _, err := net.TrainBatch(xs, ys, 0.5); err != nil {
			panic(err)
		}
	}

	for _, x := range xs {
		label, _ := net.Classify(x)
		fmt.Println(x, label)
	}
	// Output:
	// [0 0] same
	// [0 1] different
	// [1 0] different
	// [1 1] same
}
