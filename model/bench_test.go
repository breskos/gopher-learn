package model

import (
	"context"
	"math/rand/v2"
	"testing"

	"github.com/breskos/gopher-learn/nn"
)

func BenchmarkPredict(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 1))
	m, _, err := Train(context.Background(), shop(rng, 300), Config{Target: "buys", Ignore: []string{"id"}, Fit: nn.FitConfig{Epochs: 2}, Rand: rng})
	if err != nil {
		b.Fatal(err)
	}
	r := Record{"x": "0.4", "income": "52000", "color": "red"}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = m.Predict(r)
	}
}
