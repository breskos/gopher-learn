package drift_test

import (
	"fmt"
	"math/rand/v2"

	"github.com/breskos/gopher-learn/drift"
)

// Watch the error rate of a model: it is 10 % at first and jumps to 50 %.
func Example() {
	rng := rand.New(rand.NewPCG(1, 1))
	d := drift.NewADWIN(0.002)
	reported := false
	for i := range 2000 {
		rate := 0.1
		if i >= 1000 {
			rate = 0.5
		}
		e := 0.0
		if rng.Float64() < rate {
			e = 1
		}
		if d.Add(e) && d.Change > 0 && !reported {
			reported = true
			fmt.Printf("the error rate rose shortly after sample 1000: %v\n", i >= 1000 && i < 1100)
		}
	}
	fmt.Printf("current error rate: %.1f\n", d.Mean())
	// Output:
	// the error rate rose shortly after sample 1000: true
	// current error rate: 0.5
}
