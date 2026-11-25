package online

import (
	"math/rand/v2"
	"testing"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

func BenchmarkLearn(b *testing.B) {
	set := learn.NewSet(nn.Classification)
	_ = set.AddClass("A")
	_ = set.AddClass("B")
	cfg := DefaultConfig()
	cfg.Seed = 1
	o, err := New(set, 2, []int{8}, cfg)
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(1, 1))
	samples := make([]*learn.Sample, 1024)
	for i := range samples {
		samples[i] = point(rng, false)
	}
	b.ReportAllocs()
	i := 0
	for b.Loop() {
		s := *samples[i%len(samples)]
		_, _ = o.Learn(&s)
		i++
	}
}
