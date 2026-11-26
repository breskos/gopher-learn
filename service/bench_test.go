package service

import (
	"math/rand/v2"
	"testing"

	"github.com/breskos/gopher-learn/nn"
)

// BenchmarkPredict runs with a full pending store, so every prediction
// also evicts the oldest one.
func BenchmarkPredict(b *testing.B) {
	cfg := DefaultConfig()
	cfg.PendingSize = 10_000
	s, err := New(classifierNet(b, 1), nn.Classification, cfg)
	if err != nil {
		b.Fatal(err)
	}
	x := []float64{0.3, -0.2}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = s.Predict(x)
	}
}

func BenchmarkPredictParallel(b *testing.B) {
	cfg := DefaultConfig()
	cfg.PendingSize = 10_000
	s, err := New(classifierNet(b, 1), nn.Classification, cfg)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		x := []float64{0.3, -0.2}
		for pb.Next() {
			_, _ = s.Predict(x)
		}
	})
}

// BenchmarkFeedback measures one update: the new sample plus 15 replay
// samples, one clipped SGD step, and a guardrail check every 100 updates.
func BenchmarkFeedback(b *testing.B) {
	cfg := DefaultConfig()
	cfg.Seed = 1
	cfg.Reference = referenceSet(200, false)
	s, err := New(classifierNet(b, 1), nn.Classification, cfg)
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(1, 1))
	xs := make([][]float64, 1024)
	labels := make([]string, len(xs))
	for i := range xs {
		xs[i], labels[i] = concept(rng, false)
	}
	ids := make([]string, len(xs))
	b.ReportAllocs()
	i := 0
	for b.Loop() {
		k := i % len(xs)
		if k == 0 {
			b.StopTimer()
			for j, x := range xs {
				p, _ := s.Predict(x)
				ids[j] = p.ID
			}
			b.StartTimer()
		}
		_, _ = s.Feedback(ids[k], labels[k])
		i++
	}
}
