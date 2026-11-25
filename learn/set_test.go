package learn

import (
	"errors"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/breskos/gopher-learn/nn"
)

func TestReadCSVClassification(t *testing.T) {
	s := NewSet(nn.Classification)
	err := s.ReadCSV(strings.NewReader("0.1,0.2,A\n0.3,0.4,B\n0.5,0.6,A\n"))
	if err != nil {
		t.Fatal(err)
	}
	if s.Len() != 3 || s.InputSize() != 2 {
		t.Fatalf("got %d samples with %d features", s.Len(), s.InputSize())
	}
	if !slices.Equal(s.Labels, []string{"A", "B"}) {
		t.Fatalf("labels = %v", s.Labels)
	}
	if got := s.Samples[1]; got.Class != 1 || !slices.Equal(got.Output, []float64{0, 1}) {
		t.Fatalf("sample 1 = %+v", got)
	}
	// samples loaded before a new class appeared must be extended
	if got := s.Samples[0]; !slices.Equal(got.Output, []float64{1, 0}) {
		t.Fatalf("sample 0 output = %v", got.Output)
	}
}

func TestReadCSVRegression(t *testing.T) {
	s := NewSet(nn.Regression)
	if err := s.ReadCSV(strings.NewReader("1,2,0.5\n3,4,0.7\n")); err != nil {
		t.Fatal(err)
	}
	if s.Samples[1].Value != 0.7 || !slices.Equal(s.Samples[1].Output, []float64{0.7}) {
		t.Fatalf("sample = %+v", s.Samples[1])
	}
	if err := s.ReadCSV(strings.NewReader("1,2,x\n")); err == nil {
		t.Fatal("expected error for invalid target")
	}
}

func TestReadCSVErrors(t *testing.T) {
	s := NewSet(nn.Classification)
	if err := s.ReadCSV(strings.NewReader("1,abc,A\n")); err == nil {
		t.Fatal("expected parse error")
	}
	s = NewSet(nn.Classification)
	if err := s.ReadCSV(strings.NewReader("1,2,A\n1,B\n")); err == nil {
		t.Fatal("expected dimension error")
	}
}

func TestReadSVM(t *testing.T) {
	s := NewSet(nn.Classification)
	if err := s.ReadSVM(strings.NewReader("+1 1:0.5 3:1\n-1 2:0.25\n")); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(s.Samples[0].Vector, []float64{0.5, 0, 1}) {
		t.Fatalf("vector 0 = %v", s.Samples[0].Vector)
	}
	if !slices.Equal(s.Samples[1].Vector, []float64{0, 0.25, 0}) {
		t.Fatalf("vector 1 = %v", s.Samples[1].Vector)
	}
	if !slices.Equal(s.Labels, []string{"+1", "-1"}) {
		t.Fatalf("labels = %v", s.Labels)
	}
}

func TestReadSVMLimits(t *testing.T) {
	defer func(features, values int) { maxSVMFeatures, maxDenseValues = features, values }(maxSVMFeatures, maxDenseValues)
	maxSVMFeatures, maxDenseValues = 100, 250

	if err := NewSet(nn.Classification).ReadSVM(strings.NewReader("1 101:1\n")); err == nil {
		t.Fatal("expected error for feature index above the limit")
	}
	if err := NewSet(nn.Classification).ReadSVM(strings.NewReader("1 2147483647:1\n")); err == nil {
		t.Fatal("expected error for huge feature index")
	}
	// 3 rows × 100 features = 300 values > 250
	if err := NewSet(nn.Classification).ReadSVM(strings.NewReader("1 100:1\n2 1:1\n1 1:2\n")); err == nil {
		t.Fatal("expected error for too many values")
	}
	if err := NewSet(nn.Classification).ReadSVM(strings.NewReader("1 100:1\n2 1:1\n")); err != nil {
		t.Fatalf("2 rows × 100 features must be accepted: %v", err)
	}
}

func TestNonFiniteValues(t *testing.T) {
	for _, in := range []string{"1,NaN,A\n", "1,+Inf,A\n", "1,-Inf,A\n"} {
		err := NewSet(nn.Classification).ReadCSV(strings.NewReader(in))
		if !errors.Is(err, ErrNonFinite) {
			t.Errorf("ReadCSV(%q): want ErrNonFinite, got %v", in, err)
		}
	}
	if err := NewSet(nn.Regression).ReadCSV(strings.NewReader("1,2,NaN\n")); !errors.Is(err, ErrNonFinite) {
		t.Errorf("NaN target: want ErrNonFinite, got %v", err)
	}
	if err := NewSet(nn.Classification).ReadSVM(strings.NewReader("1 1:NaN\n")); !errors.Is(err, ErrNonFinite) {
		t.Errorf("ReadSVM NaN: want ErrNonFinite, got %v", err)
	}
}

func TestEmptyFeatures(t *testing.T) {
	if err := NewSet(nn.Classification).ReadSVM(strings.NewReader("1\n")); err == nil {
		t.Fatal("expected error for a sample without features")
	}
}

func TestAddUnknownClass(t *testing.T) {
	s := NewSet(nn.Classification)
	if err := s.Add(NewClassificationSample([]float64{1}, "X")); !errors.Is(err, ErrUnknownClass) {
		t.Fatalf("want ErrUnknownClass, got %v", err)
	}
	if err := NewSet(nn.Regression).AddClass("A"); !errors.Is(err, ErrTaskMismatch) {
		t.Fatalf("want ErrTaskMismatch, got %v", err)
	}
}

func TestContains(t *testing.T) {
	s := NewSet(nn.Regression)
	_ = s.Add(NewRegressionSample([]float64{1, 2}, 0.5))
	if !s.Contains(NewRegressionSample([]float64{1, 2}, 0.9)) {
		t.Fatal("expected duplicate vector to be found")
	}
	if s.Contains(NewRegressionSample([]float64{2, 1}, 0.5)) {
		t.Fatal("unexpected match")
	}
}

func TestSplit(t *testing.T) {
	s := NewSet(nn.Regression)
	for i := range 10 {
		_ = s.Add(NewRegressionSample([]float64{float64(i)}, 0))
	}
	a, b, err := s.Split(0.7, rand.New(rand.NewPCG(1, 1)))
	if err != nil {
		t.Fatal(err)
	}
	if a.Len() != 7 || b.Len() != 3 {
		t.Fatalf("split = %d/%d", a.Len(), b.Len())
	}
	if _, _, err := s.Split(1, nil); err == nil {
		t.Fatal("expected error for ratio 1")
	}
}

func TestReplaceAndDropOldest(t *testing.T) {
	s := NewSet(nn.Regression)
	a, b, c := NewRegressionSample([]float64{1}, 1), NewRegressionSample([]float64{2}, 2), NewRegressionSample([]float64{3}, 3)
	for _, sm := range []*Sample{a, b, a, c} {
		if err := s.Add(sm); err != nil {
			t.Fatal(err)
		}
	}
	s.DropOldest(1)
	if s.Len() != 3 || !s.Contains(a) {
		t.Fatal("the duplicate of the dropped sample must stay known")
	}
	s.DropOldest(2)
	if s.Len() != 1 || s.Contains(a) || s.Contains(b) || !s.Contains(c) {
		t.Fatalf("after dropping: %d samples", s.Len())
	}
	if err := s.Replace(0, NewRegressionSample([]float64{4}, 4)); err != nil {
		t.Fatal(err)
	}
	if s.Contains(c) || s.Samples[0].Output[0] != 4 {
		t.Fatal("replaced sample still indexed")
	}
	if err := s.Replace(1, a); err == nil {
		t.Error("out of range: expected error")
	}
	if err := s.Replace(0, NewRegressionSample([]float64{1, 2}, 1)); err == nil {
		t.Error("wrong dimension: expected error")
	}
	s.DropOldest(10)
	if s.Len() != 0 {
		t.Fatal("drop more than the size")
	}
	s.DropOldest(-1)
}
