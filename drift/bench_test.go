package drift

import (
	"math/rand/v2"
	"testing"
)

func BenchmarkADWIN(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 1))
	a := NewADWIN(0.002)
	b.ReportAllocs()
	for b.Loop() {
		a.Add(bernoulli(rng, 0.2))
	}
}

func BenchmarkPageHinkley(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 1))
	p := NewPageHinkley()
	b.ReportAllocs()
	for b.Loop() {
		p.Add(bernoulli(rng, 0.2))
	}
}
