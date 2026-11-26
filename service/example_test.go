package service_test

import (
	"fmt"
	"math/rand/v2"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/service"
)

func Example() {
	// A trained (here: untrained) network to start with.
	net, _ := nn.New(nn.Config{
		Inputs: 2, Hidden: []int{8}, Outputs: 2, Labels: []string{"buy", "skip"},
		HiddenActivation: nn.Tanh, OutputActivation: nn.Softmax,
		Rand: rand.New(rand.NewPCG(1, 1)),
	})

	// A stable reference set the shadow network must do well on before it
	// goes live.
	rng := rand.New(rand.NewPCG(2, 2))
	label := func(x []float64) string {
		if x[0]+x[1] > 0 {
			return "buy"
		}
		return "skip"
	}
	ref := learn.NewSet(nn.Classification)
	_ = ref.AddClass("buy")
	_ = ref.AddClass("skip")
	for range 200 {
		x := []float64{rng.NormFloat64(), rng.NormFloat64()}
		_ = ref.Add(learn.NewClassificationSample(x, label(x)))
	}

	cfg := service.DefaultConfig()
	cfg.Reference = ref
	cfg.Seed = 3
	s, err := service.New(net, nn.Classification, cfg)
	if err != nil {
		panic(err)
	}

	for range 2000 {
		// A request: predict live, remember the event ID.
		x := []float64{rng.NormFloat64(), rng.NormFloat64()}
		p, err := s.Predict(x)
		if err != nil {
			panic(err)
		}
		// ... later, the outcome is known.
		if _, err := s.Feedback(p.ID, label(x)); err != nil {
			panic(err)
		}
	}
	st := s.Stats()
	fmt.Printf("version %d, %d updates, live accuracy above 95 %%: %v\n", st.Version, st.Updates, st.Accuracy > 0.95)
	// Output: version 21, 2000 updates, live accuracy above 95 %: true
}
