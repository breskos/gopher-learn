// Command online-generator feeds generated data points into an online learner.
// It needs no data files.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math/rand/v2"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/online"
)

var centers = map[string][]float64{
	"Y": {1.0, 3.0, 10.5, 5.0, 4.0, 3.3, 5.2},
	"N": {1.0, 8.7, 1.3, 3.3, 4.0, 10.1, 5.1},
}

func main() {
	set := learn.NewSet(nn.Classification)
	for _, label := range []string{"Y", "N"} {
		if err := set.AddClass(label); err != nil {
			log.Fatal(err)
		}
	}
	o, err := online.New(set, len(centers["Y"]), []int{30}, online.DefaultConfig())
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	for i := range 2000 {
		label := set.Labels[rand.IntN(len(set.Labels))]
		sample := learn.NewClassificationSample(generate(label), label)
		if err := o.Inject(sample, false); err != nil && !errors.Is(err, online.ErrSampleExists) {
			log.Fatal(err)
		}
		if i%200 == 199 {
			score, err := o.Iterate(ctx)
			if err != nil {
				log.Fatal(err)
			}
			fmt.Printf("after %4d samples: F1 %.3f\n", i+1, score)
		}
	}
}

// generate returns a noisy point around the center of the class.
func generate(label string) []float64 {
	c := centers[label]
	v := make([]float64, len(c))
	for i, x := range c {
		v[i] = (x - 1) + rand.Float64()*(x+1)
	}
	return v
}
