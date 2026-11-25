package learn

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/breskos/gopher-learn/nn"
)

// unbalanced returns a set with 90 samples of "a" and 10 of "b".
func unbalanced(t *testing.T) *Set {
	t.Helper()
	s := NewSet(nn.Classification)
	_ = s.AddClass("a")
	_ = s.AddClass("b")
	for i := range 100 {
		label := "a"
		if i%10 == 0 {
			label = "b"
		}
		if err := s.Add(NewClassificationSample([]float64{float64(i)}, label)); err != nil {
			t.Fatal(err)
		}
	}
	return s
}

func TestStratifiedSplit(t *testing.T) {
	s := unbalanced(t)
	train, test, err := s.StratifiedSplit(0.8, rand.New(rand.NewPCG(1, 1)))
	if err != nil {
		t.Fatal(err)
	}
	if got := train.Distribution(); got["a"] != 72 || got["b"] != 8 {
		t.Errorf("train distribution = %v", got)
	}
	if got := test.Distribution(); got["a"] != 18 || got["b"] != 2 {
		t.Errorf("test distribution = %v", got)
	}
	seen := map[*Sample]bool{}
	for _, part := range []*Set{train, test} {
		for _, sm := range part.Samples {
			if seen[sm] {
				t.Fatal("sample in both parts")
			}
			seen[sm] = true
		}
	}
	if len(seen) != 100 {
		t.Fatalf("%d of 100 samples assigned", len(seen))
	}
}

func TestStratifiedSplitRareClass(t *testing.T) {
	s := NewSet(nn.Classification)
	_ = s.AddClass("common")
	_ = s.AddClass("rare")
	for i := range 20 {
		_ = s.Add(NewClassificationSample([]float64{float64(i)}, "common"))
	}
	_ = s.Add(NewClassificationSample([]float64{100}, "rare"))
	_ = s.Add(NewClassificationSample([]float64{101}, "rare"))
	train, test, err := s.StratifiedSplit(0.9, nil)
	if err != nil {
		t.Fatal(err)
	}
	if train.Distribution()["rare"] != 1 || test.Distribution()["rare"] != 1 {
		t.Fatalf("rare class must appear in both parts: %v / %v", train.Distribution(), test.Distribution())
	}
}

func TestStratifiedSplitRegressionAndErrors(t *testing.T) {
	s := regressionSet(t, []float64{1}, []float64{2}, []float64{3}, []float64{4})
	a, b, err := s.StratifiedSplit(0.5, rand.New(rand.NewPCG(1, 1)))
	if err != nil || a.Len() != 2 || b.Len() != 2 {
		t.Fatalf("regression split: %d/%d (%v)", a.Len(), b.Len(), err)
	}
	if _, _, err := unbalanced(t).StratifiedSplit(0, nil); err == nil {
		t.Fatal("expected error for ratio 0")
	}
}

func TestKFold(t *testing.T) {
	s := unbalanced(t)
	folds, err := s.KFold(5, rand.New(rand.NewPCG(2, 2)))
	if err != nil {
		t.Fatal(err)
	}
	if len(folds) != 5 {
		t.Fatalf("%d folds", len(folds))
	}
	tested := map[*Sample]int{}
	for i, f := range folds {
		if f.Train.Len()+f.Test.Len() != 100 {
			t.Errorf("fold %d: %d + %d samples", i, f.Train.Len(), f.Test.Len())
		}
		if d := f.Test.Distribution(); d["a"] != 18 || d["b"] != 2 {
			t.Errorf("fold %d not stratified: %v", i, d)
		}
		inTrain := map[*Sample]bool{}
		for _, sm := range f.Train.Samples {
			inTrain[sm] = true
		}
		for _, sm := range f.Test.Samples {
			if inTrain[sm] {
				t.Fatalf("fold %d: sample in train and test", i)
			}
			tested[sm]++
		}
	}
	for sm, n := range tested {
		if n != 1 {
			t.Fatalf("sample %v tested %d times", sm.Vector, n)
		}
	}
	if len(tested) != 100 {
		t.Fatalf("%d samples tested", len(tested))
	}
}

func TestKFoldErrors(t *testing.T) {
	s := regressionSet(t, []float64{1}, []float64{2}, []float64{3})
	for _, k := range []int{1, 4} {
		if _, err := s.KFold(k, nil); err == nil {
			t.Errorf("k=%d: expected error", k)
		}
	}
	folds, err := s.KFold(3, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, f := range folds {
		if f.Test.Len() != 1 || f.Train.Len() != 2 {
			t.Errorf("fold %d: %s", i, fmt.Sprint(f.Train.Len(), "/", f.Test.Len()))
		}
	}
}
