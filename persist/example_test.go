package persist_test

import (
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"

	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/persist"
)

func Example() {
	net, err := nn.New(nn.Config{
		Inputs: 3,
		Hidden: []int{4},
		Labels: []string{"no", "yes"},
		Rand:   rand.New(rand.NewPCG(1, 1)),
	})
	if err != nil {
		panic(err)
	}

	dir, err := os.MkdirTemp("", "persist-example")
	if err != nil {
		panic(err)
	}
	defer func() { _ = os.RemoveAll(dir) }()
	path := filepath.Join(dir, "network.json")

	if err := persist.SaveNetwork(path, net); err != nil {
		panic(err)
	}
	loaded, err := persist.LoadNetwork(path)
	if err != nil {
		panic(err)
	}

	x := []float64{0.1, 0.2, 0.3}
	a, _ := net.Predict(x)
	b, _ := loaded.Predict(x)
	fmt.Println("same predictions:", slices.Equal(a, b))
	fmt.Println("labels:", loaded.Labels)
	// Output:
	// same predictions: true
	// labels: [no yes]
}
