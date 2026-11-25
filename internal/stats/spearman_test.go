package stats

import (
	"math"
	"testing"
)

func TestSpearman(t *testing.T) {
	tests := []struct {
		x, y []float64
		want float64
	}{
		{[]float64{1, 2, 3, 4}, []float64{10, 20, 30, 40}, 1},
		{[]float64{1, 2, 3, 4}, []float64{4, 3, 2, 1}, -1},
		{[]float64{1, 2, 3, 4, 5}, []float64{1, 4, 9, 16, 25}, 1},
		{[]float64{1, 2, 2, 3}, []float64{1, 2, 3, 4}, 0.9486832980505138},
	}
	for _, tt := range tests {
		if got := Spearman(tt.x, tt.y); math.Abs(got-tt.want) > 1e-12 {
			t.Errorf("Spearman(%v, %v) = %v, want %v", tt.x, tt.y, got, tt.want)
		}
	}
	if !math.IsNaN(Spearman([]float64{1, 1, 1}, []float64{1, 2, 3})) {
		t.Error("constant input must yield NaN")
	}
}

func TestSpearmanInvalidInput(t *testing.T) {
	if !math.IsNaN(Spearman([]float64{1, 2}, []float64{1})) {
		t.Error("different lengths must yield NaN")
	}
	if !math.IsNaN(Spearman([]float64{1}, []float64{1})) {
		t.Error("a single value must yield NaN")
	}
}
