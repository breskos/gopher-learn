package pca

import (
	"math/rand/v2"
	"testing"
)

func points(n, d int) [][]float64 {
	rng := rand.New(rand.NewPCG(1, 1))
	p := make([][]float64, n)
	for i := range p {
		p[i] = make([]float64, d)
		for j := range p[i] {
			p[i][j] = rng.NormFloat64()
		}
	}
	return p
}

func BenchmarkFit100(b *testing.B) {
	p := points(10000, 100)
	for b.Loop() {
		_, _ = Fit(p, Config{Components: 10})
	}
}

func BenchmarkFit784(b *testing.B) {
	p := points(2000, 784)
	for b.Loop() {
		_, _ = Fit(p, Config{Components: 50})
	}
}

func BenchmarkTransformInto(b *testing.B) {
	p := points(1000, 100)
	m, _ := Fit(p, Config{Components: 10})
	z := make([]float64, 10)
	b.ReportAllocs()
	for b.Loop() {
		_ = m.TransformInto(z, p[0])
	}
}
