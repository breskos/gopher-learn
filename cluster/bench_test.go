package cluster

import (
	"math/rand/v2"
	"testing"
)

func BenchmarkFit(b *testing.B) {
	points, _ := gaussians(10000, 8, 10, 1)
	for b.Loop() {
		_, _, _ = Fit(points, Config{K: 8, Rand: rand.New(rand.NewPCG(1, 1))})
	}
}

func BenchmarkPredict(b *testing.B) {
	points, _ := gaussians(1000, 8, 10, 1)
	m, _, _ := Fit(points, Config{K: 8, Restarts: 1})
	b.ReportAllocs()
	for b.Loop() {
		_, _ = m.Predict(points[0])
	}
}
