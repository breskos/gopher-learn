package learn

import (
	"encoding/json"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/breskos/gopher-learn/nn"
)

func regressionSet(t *testing.T, vectors ...[]float64) *Set {
	t.Helper()
	s := NewSet(nn.Regression)
	for _, v := range vectors {
		if err := s.Add(NewRegressionSample(v, 0.5)); err != nil {
			t.Fatal(err)
		}
	}
	return s
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestStandardize(t *testing.T) {
	s := regressionSet(t, []float64{1, 10, 5}, []float64{2, 20, 5}, []float64{3, 30, 5})
	sc, err := FitScaler(s, Standardize)
	if err != nil {
		t.Fatal(err)
	}
	scaled, err := sc.TransformSet(s)
	if err != nil {
		t.Fatal(err)
	}
	for f := range 2 {
		var mean, sq float64
		for _, sm := range scaled.Samples {
			mean += sm.Vector[f]
		}
		mean /= 3
		for _, sm := range scaled.Samples {
			sq += (sm.Vector[f] - mean) * (sm.Vector[f] - mean)
		}
		if !near(mean, 0) || !near(sq/3, 1) {
			t.Errorf("feature %d: mean %v, variance %v", f, mean, sq/3)
		}
	}
	// constant feature is shifted to 0, not divided by zero
	if v := scaled.Samples[0].Vector[2]; v != 0 {
		t.Errorf("constant feature scaled to %v", v)
	}
	// the original set is untouched
	if !slices.Equal(s.Samples[0].Vector, []float64{1, 10, 5}) {
		t.Errorf("original modified: %v", s.Samples[0].Vector)
	}
}

func TestMinMax(t *testing.T) {
	s := regressionSet(t, []float64{-1, 100}, []float64{3, 200}, []float64{1, 150})
	sc, err := FitScaler(s, MinMax)
	if err != nil {
		t.Fatal(err)
	}
	v, _ := sc.Transform([]float64{3, 100})
	if !near(v[0], 1) || !near(v[1], 0) {
		t.Fatalf("transform = %v", v)
	}
	back, err := sc.Inverse(v)
	if err != nil || !near(back[0], 3) || !near(back[1], 100) {
		t.Fatalf("inverse = %v (%v)", back, err)
	}
	// values outside the training range are not clipped
	v, _ = sc.Transform([]float64{5, 300})
	if !near(v[0], 1.5) || !near(v[1], 2) {
		t.Fatalf("extrapolation = %v", v)
	}
}

func TestScalerErrors(t *testing.T) {
	if _, err := FitScaler(NewSet(nn.Regression), Standardize); err == nil {
		t.Error("expected error for empty set")
	}
	if _, err := FitScaler(nil, MinMax); err == nil {
		t.Error("expected error for nil set")
	}
	s := regressionSet(t, []float64{1, 2})
	if _, err := FitScaler(s, ScaleMethod(7)); err == nil {
		t.Error("expected error for invalid method")
	}
	sc, _ := FitScaler(s, MinMax)
	if _, err := sc.Transform([]float64{1}); !errors.Is(err, nn.ErrDimension) {
		t.Errorf("Transform: want ErrDimension, got %v", err)
	}
	if _, err := sc.Inverse([]float64{1}); !errors.Is(err, nn.ErrDimension) {
		t.Errorf("Inverse: want ErrDimension, got %v", err)
	}
	other := regressionSet(t, []float64{1, 2, 3})
	if _, err := sc.TransformSet(other); err == nil {
		t.Error("TransformSet: expected dimension error")
	}
}

func TestScalerJSONAndValidate(t *testing.T) {
	s := regressionSet(t, []float64{1, 2}, []float64{3, 6})
	sc, _ := FitScaler(s, Standardize)
	b, err := json.Marshal(sc)
	if err != nil {
		t.Fatal(err)
	}
	var back Scaler
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if err := back.Validate(); err != nil {
		t.Fatal(err)
	}
	if back.Method != Standardize || !slices.Equal(back.Factor, sc.Factor) {
		t.Fatalf("round trip = %+v", back)
	}
	for name, bad := range map[string]Scaler{
		"empty":    {},
		"lengths":  {Offset: []float64{1}, Factor: []float64{1, 2}},
		"zero":     {Offset: []float64{1}, Factor: []float64{0}},
		"infinite": {Offset: []float64{math.Inf(1)}, Factor: []float64{1}},
	} {
		if err := bad.Validate(); err == nil {
			t.Errorf("%s: expected validation error", name)
		}
	}
	var m ScaleMethod
	if err := m.UnmarshalText([]byte("robust")); err == nil {
		t.Error("expected error for unknown method")
	}
	if _, err := ScaleMethod(9).MarshalText(); err == nil {
		t.Error("expected error for invalid method")
	}
	if ScaleMethod(9).String() != "ScaleMethod(9)" || MinMax.String() != "min-max" {
		t.Error("unexpected method names")
	}
}
