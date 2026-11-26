package linear

import "testing"

func BenchmarkFitRegression(b *testing.B) {
	s := exact(10000, 0.5, 1)
	for b.Loop() {
		_, _ = FitRegression(s, RegressionConfig{})
	}
}

func BenchmarkFitLogistic(b *testing.B) {
	s := blobs(10000, 4, 20, 2, 1)
	for b.Loop() {
		_, _ = FitLogistic(s, LogisticConfig{})
	}
}

func BenchmarkProbabilitiesInto(b *testing.B) {
	s := blobs(1000, 4, 20, 2, 1)
	m, _ := FitLogistic(s, LogisticConfig{})
	p := make([]float64, 4)
	x := s.Samples[0].Vector
	b.ReportAllocs()
	for b.Loop() {
		_ = m.ProbabilitiesInto(p, x)
	}
}
