package synth_test

import (
	"context"
	"testing"

	"github.com/breskos/gopher-learn/internal/synth"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/online"
)

// learnStream feeds a stream to Learn row by row (test-then-train),
// retraining every 250 rows, and returns the per-row errors and the rows at
// which drift was detected.
func learnStream(t *testing.T, path string, rows int, cfg online.Config) ([]float64, []int) {
	t.Helper()
	d, err := synth.ByPath(path)
	if err != nil {
		t.Fatal(err)
	}
	data, err := d.Set(rows, 42)
	if err != nil {
		t.Fatal(err)
	}
	set := learn.NewSet(data.Task)
	for _, l := range data.Labels {
		_ = set.AddClass(l)
	}
	cfg.Seed = 1
	o, err := online.New(set, data.InputSize(), []int{12}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	var errs []float64
	var drifts []int
	for i, s := range data.Samples {
		c := *s
		r, err := o.Learn(&c)
		if err != nil {
			t.Fatal(err)
		}
		errs = append(errs, r.Error)
		if r.Drift {
			drifts = append(drifts, i)
		}
		if (i+1)%250 == 0 && o.Data.Len() >= cfg.MinimumDataPoints {
			if _, err := o.Iterate(context.Background()); err != nil {
				t.Fatal(err)
			}
		}
	}
	return errs, drifts
}

func meanAccuracy(errs []float64) float64 {
	var sum float64
	for _, e := range errs {
		sum += e
	}
	return 1 - sum/float64(len(errs))
}

// TestLearnRecoversFromDrift compares the prequential accuracy after a
// drift with and without drift detection.
func TestLearnRecoversFromDrift(t *testing.T) {
	if testing.Short() {
		t.Skip("quality tests train real models")
	}
	cases := []struct {
		path          string
		rows          int
		quiet         int    // no detection is expected before this row
		detect        [2]int // rows in which the drift must be detected
		before, after [2]int // evaluation windows
		beatStatic    bool   // detection must clearly beat a learner without it
	}{
		{"online/abrupt-drift.csv", 8000, 3500, [2]int{4000, 4500}, [2]int{3000, 4000}, [2]int{4500, 6000}, true},
		// concept 2 takes over between rows 3200 and 4800
		{"online/gradual-drift.csv", 8000, 2700, [2]int{3200, 5000}, [2]int{2200, 3200}, [2]int{5300, 6800}, true},
		// concepts switch every 2000 rows; the fast hot-shot updates recover
		// without detection too, so only the detection itself is checked
		{"online/recurring-concepts.csv", 16000, 0, [2]int{4000, 4500}, [2]int{3000, 4000}, [2]int{4500, 6000}, false},
	}
	for _, c := range cases {
		t.Run(c.path, func(t *testing.T) {
			adaptive := online.DefaultConfig()
			adaptive.MemorySize = 2000
			errs, drifts := learnStream(t, c.path, c.rows, adaptive)

			static := online.DefaultConfig()
			static.Memory, static.Drift = online.Unbounded, online.NoDrift
			staticErrs, _ := learnStream(t, c.path, c.rows, static)

			before := meanAccuracy(errs[c.before[0]:c.before[1]])
			after := meanAccuracy(errs[c.after[0]:c.after[1]])
			staticAfter := meanAccuracy(staticErrs[c.after[0]:c.after[1]])
			t.Logf("drift detected at %v; prequential accuracy before %.3f, after %.3f (without detection %.3f)", drifts, before, after, staticAfter)

			if before < 0.85 {
				t.Errorf("accuracy before the drift %.3f", before)
			}
			detected := false
			for _, d := range drifts {
				if d >= c.detect[0] && d < c.detect[1] {
					detected = true
				}
				if d < c.quiet {
					t.Errorf("false alarm at row %d", d)
				}
			}
			if !detected {
				t.Errorf("drift not detected in rows %v: %v", c.detect, drifts)
			}
			if after < 0.85 || (c.beatStatic && after < staticAfter+0.05) {
				t.Errorf("accuracy after the drift %.3f, without detection %.3f", after, staticAfter)
			}
		})
	}
}
