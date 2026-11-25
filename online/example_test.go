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

// Learn from a stream whose concept changes after 2000 samples: the label
// first depends on x1 + x2, then on x1 - x2. The drift is detected and the
// learner forgets the samples of the old concept.
func ExampleOnline_Learn() {
	set := learn.NewSet(nn.Classification)
	_ = set.AddClass("A")
	_ = set.AddClass("B")
	cfg := online.DefaultConfig()
	cfg.Seed = 1
	cfg.OnDrift = func(e online.DriftEvent) {
		fmt.Printf("drift detected shortly after the change: %v\n", e.Sample > 2000 && e.Sample < 2100)
	}
	o, err := online.New(set, 2, []int{12}, cfg)
	if err != nil {
		panic(err)
	}
	rng := rand.New(rand.NewPCG(2, 2))
	for i := range 4000 {
		x := []float64{rng.NormFloat64(), rng.NormFloat64()}
		score := x[0] + x[1]
		if i >= 2000 {
			score = x[0] - x[1]
		}
		label := "A"
		if score <= 0 {
			label = "B"
		}
		if _, err := o.Learn(learn.NewClassificationSample(x, label)); err != nil {
			panic(err)
		}
		if (i+1)%250 == 0 {
			if _, err := o.Iterate(context.Background()); err != nil {
				panic(err)
			}
		}
	}
	fmt.Printf("accuracy on the last %d samples above 0.9: %v\n", o.Prequential.Len(), o.Prequential.Accuracy() > 0.9)
	// Output:
	// drift detected shortly after the change: true
	// accuracy on the last 500 samples above 0.9: true
}
