package online

import (
	"context"
	"errors"
	"math/rand/v2"
	"testing"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

func TestOnlineLearnsStream(t *testing.T) {
	set := learn.NewSet(nn.Classification)
	_ = set.AddClass("Y")
	_ = set.AddClass("N")
	cfg := DefaultConfig()
	cfg.Seed = 9
	o, err := New(set, 2, []int{6}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if _, err := o.Iterate(ctx); !errors.Is(err, ErrNotEnoughData) {
		t.Fatalf("want ErrNotEnoughData, got %v", err)
	}

	rng := rand.New(rand.NewPCG(5, 5))
	var score float64
	for i := range 300 {
		label, c := "Y", 1.0
		if rng.IntN(2) == 0 {
			label, c = "N", -1.0
		}
		s := learn.NewClassificationSample([]float64{c + rng.NormFloat64()*0.2, -c + rng.NormFloat64()*0.2}, label)
		if err := o.Inject(s, false); err != nil {
			t.Fatal(err)
		}
		if i%50 == 49 {
			if score, err = o.Iterate(ctx); err != nil {
				t.Fatal(err)
			}
		}
	}
	if score < 0.9 {
		t.Fatalf("score = %.3f, want >= 0.9", score)
	}
	if err := o.Inject(o.Data.Samples[0], false); !errors.Is(err, ErrSampleExists) {
		t.Fatalf("want ErrSampleExists, got %v", err)
	}
}
