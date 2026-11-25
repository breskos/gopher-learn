package learn

import (
	"math"
	"strings"
	"testing"

	"github.com/breskos/gopher-learn/nn"
)

// checkSet verifies the invariants every successfully loaded set must hold.
func checkSet(t *testing.T, s *Set) {
	t.Helper()
	for i, sm := range s.Samples {
		if len(sm.Vector) != s.InputSize() {
			t.Fatalf("sample %d has %d features, set has %d", i, len(sm.Vector), s.InputSize())
		}
		if len(sm.Output) != s.OutputSize() {
			t.Fatalf("sample %d has output %v, set output size %d", i, sm.Output, s.OutputSize())
		}
		for _, v := range sm.Vector {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("sample %d contains non-finite value %v", i, v)
			}
		}
		if s.Task == nn.Classification {
			if sm.Class < 0 || sm.Class >= len(s.Labels) || s.Labels[sm.Class] != sm.Label {
				t.Fatalf("sample %d has class %d for label %q, labels %v", i, sm.Class, sm.Label, s.Labels)
			}
			if sm.Output[sm.Class] != 1 {
				t.Fatalf("sample %d output %v is not one-hot at %d", i, sm.Output, sm.Class)
			}
		}
	}
}

func FuzzReadCSV(f *testing.F) {
	for _, seed := range []string{
		"0.1,0.2,A\n0.3,0.4,B\n",
		"1,2,0.5\n3,4,0.7\n",
		"1,NaN,A\n",
		"1,+Inf,A\n",
		"\"1\",\"2\",\"x\"\n",
		"1,2\n1,2,3\n",
		"",
		",\n",
	} {
		f.Add(seed, false)
		f.Add(seed, true)
	}
	f.Fuzz(func(t *testing.T, data string, regression bool) {
		task := nn.Classification
		if regression {
			task = nn.Regression
		}
		s := NewSet(task)
		if err := s.ReadCSV(strings.NewReader(data)); err != nil {
			return
		}
		checkSet(t, s)
	})
}

func FuzzReadSVM(f *testing.F) {
	for _, seed := range []string{
		"+1 1:0.5 3:1\n-1 2:0.25\n",
		"0.5 1:1 2:2\n",
		"# comment\n\n1 1:1\n",
		"1 999999999999:1\n",
		"1 2147483647:1\n",
		"1 0:1\n",
		"1 1:NaN\n",
		"1 1:1:1\n",
	} {
		f.Add(seed, false)
		f.Add(seed, true)
	}
	// Keep allocations tiny while fuzzing: at most 64K values (512 KiB) per input.
	defer func(features, values int) { maxSVMFeatures, maxDenseValues = features, values }(maxSVMFeatures, maxDenseValues)
	maxSVMFeatures, maxDenseValues = 1<<12, 1<<16
	f.Fuzz(func(t *testing.T, data string, regression bool) {
		task := nn.Classification
		if regression {
			task = nn.Regression
		}
		s := NewSet(task)
		if err := s.ReadSVM(strings.NewReader(data)); err != nil {
			return
		}
		checkSet(t, s)
	})
}
