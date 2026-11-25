package engine_test

import (
	"context"
	"fmt"
	"math/rand/v2"

	"github.com/breskos/gopher-learn/engine"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

func Example() {
	// Two well separated clusters.
	rng := rand.New(rand.NewPCG(1, 1))
	data := learn.NewSet(nn.Classification)
	_ = data.AddClass("left")
	_ = data.AddClass("right")
	for i := range 100 {
		label, x := "left", -1.0
		if i%2 == 1 {
			label, x = "right", 1.0
		}
		v := []float64{x + rng.NormFloat64()*0.2, rng.NormFloat64() * 0.2}
		_ = data.Add(learn.NewClassificationSample(v, label))
	}

	cfg := engine.DefaultConfig()
	cfg.Epochs = 5
	cfg.Decay = 0.1
	cfg.Seed = 42 // reproducible
	e, err := engine.New(data, []int{4}, cfg)
	if err != nil {
		panic(err)
	}
	if err := e.Run(context.Background(), engine.FMeasure); err != nil {
		panic(err)
	}

	network, eval := e.Winner()
	label, _ := network.Classify([]float64{0.9, 0.1})
	fmt.Println("prediction:", label)
	fmt.Println("validation F1 >= 0.9:", eval.OverallFMeasure() >= 0.9)
	// Output:
	// prediction: right
	// validation F1 >= 0.9: true
}
