package online_test

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/online"
)

func Example() {
	set := learn.NewSet(nn.Classification)
	_ = set.AddClass("up")
	_ = set.AddClass("down")

	cfg := online.DefaultConfig()
	cfg.Seed = 7 // reproducible
	o, err := online.New(set, 2, []int{6}, cfg)
	if err != nil {
		panic(err)
	}

	// Samples arrive one by one, e.g. from a message queue.
	rng := rand.New(rand.NewPCG(3, 3))
	ctx := context.Background()
	var score float64
	for i := range 200 {
		label, y := "up", 1.0
		if rng.IntN(2) == 0 {
			label, y = "down", -1.0
		}
		s := learn.NewClassificationSample([]float64{rng.NormFloat64() * 0.2, y + rng.NormFloat64()*0.2}, label)
		if err := o.Inject(s, false); err != nil && !errors.Is(err, online.ErrSampleExists) {
			panic(err)
		}
		if i%50 == 49 {
			if score, err = o.Iterate(ctx); err != nil {
				panic(err)
			}
		}
	}
	fmt.Println("F1 >= 0.9:", score >= 0.9)
	// Output:
	// F1 >= 0.9: true
}
