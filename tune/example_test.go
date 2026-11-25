package tune_test

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/tune"
)

// Find out how many hidden neurons it takes to separate two rings.
func Example() {
	rng := rand.New(rand.NewPCG(1, 1))
	data := learn.NewSet(nn.Classification)
	_ = data.AddClass("inner")
	_ = data.AddClass("outer")
	for i := range 400 {
		label, r := "inner", 0.5
		if i%2 == 1 {
			label, r = "outer", 1.5
		}
		a := rng.Float64() * 2 * math.Pi
		_ = data.Add(learn.NewClassificationSample([]float64{r * math.Cos(a), r * math.Sin(a)}, label))
	}

	space := tune.Space{Hidden: [][]int{{1}, {16}}, LearningRate: []float64{0.03}}
	res, err := tune.Search(context.Background(), data, space.Grid(), tune.Config{Folds: 4, Metric: evaluation.MacroF1, Seed: 1})
	if err != nil {
		panic(err)
	}
	best := res.Best()
	fmt.Println("best hidden layers:", best.Candidate.Hidden)
	fmt.Println("F1 above 0.95:", best.Mean > 0.95)

	network, scaler, err := tune.Fit(context.Background(), data, best.Candidate, tune.Config{Seed: 1})
	if err != nil {
		panic(err)
	}
	x, _ := scaler.Transform([]float64{0, 1.5})
	label, _ := network.Classify(x)
	fmt.Println("(0, 1.5) is", label)
	// Output:
	// best hidden layers: [16]
	// F1 above 0.95: true
	// (0, 1.5) is outer
}
