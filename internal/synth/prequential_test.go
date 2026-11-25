package synth_test

import (
	"context"
	"testing"

	"github.com/breskos/gopher-learn/internal/synth"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/online"
)

// learnStream feeds a stream to Learn row by row (test-then-train),
// retraining every 250 rows, and returns the per-row errors and the rows at
// which drift was detected. If scaleRows is positive, the features are
// standardized with a scaler fitted on that many first rows.
func learnStream(t *testing.T, path string, rows, scaleRows int, cfg online.Config) ([]float64, []int) {
	t.Helper()
	d, err := synth.ByPath(path)
	if err != nil {
		t.Fatal(err)
	}
	data, err := d.Set(rows, 42)
	if err != nil {
		t.Fatal(err)
	}
	if scaleRows > 0 {
		sc, err := learn.FitScaler(&learn.Set{Task: data.Task, Samples: data.Samples[:scaleRows]}, learn.Standardize)
		if err != nil {
			t.Fatal(err)
		}
		if data, err = sc.TransformSet(data); err != nil {
			t.Fatal(err)
		}
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
			errs, drifts := learnStream(t, c.path, c.rows, 0, adaptive)

			static := online.DefaultConfig()
			static.Memory, static.Drift = online.Unbounded, online.NoDrift
			staticErrs, _ := learnStream(t, c.path, c.rows, 0, static)

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

// regressionConfig is an online configuration for regression on scaled data.
func regressionConfig() online.Config {
	cfg := online.DefaultConfig()
	cfg.HiddenActivation, cfg.OutputActivation = nn.Tanh, nn.Linear
	cfg.HotShotRate = 0.05
	cfg.MinLearningRate, cfg.MaxLearningRate = 0.01, 0.05
	return cfg
}

func meanError(errs []float64) float64 { return 1 - meanAccuracy(errs) }

// TestLearnRegressionDrift compares the prequential mean absolute error of
// regression streams with and without drift detection and bounded memory.
func TestLearnRegressionDrift(t *testing.T) {
	if testing.Short() {
		t.Skip("quality tests train real models")
	}
	cases := []struct {
		path     string
		rows     int
		detect   [2]int // rows in which a drift must be detected; zero for none
		after    [2]int // evaluation window
		maxRatio float64
	}{
		// the coefficients drift slowly; the window forgets outdated samples
		{"online/incremental-drift.csv", 10000, [2]int{}, [2]int{7500, 10000}, 0.95},
		// the sensor is replaced at two thirds of the stream (row 17520)
		{"online/air-quality-aging.csv", 26280, [2]int{17520, 19520}, [2]int{19520, 26280}, 0.9},
	}
	for _, c := range cases {
		t.Run(c.path, func(t *testing.T) {
			errs, drifts := learnStream(t, c.path, c.rows, 1000, regressionConfig())
			static := regressionConfig()
			static.Memory, static.Drift = online.Unbounded, online.NoDrift
			staticErrs, _ := learnStream(t, c.path, c.rows, 1000, static)

			mae := meanError(errs[c.after[0]:c.after[1]])
			staticMAE := meanError(staticErrs[c.after[0]:c.after[1]])
			t.Logf("drifts at %v; prequential MAE %.4f, without detection %.4f", drifts, mae, staticMAE)
			if mae > c.maxRatio*staticMAE {
				t.Errorf("MAE %.4f is not below %.2f × %.4f", mae, c.maxRatio, staticMAE)
			}
			if c.detect != [2]int{} {
				found := false
				for _, d := range drifts {
					found = found || d >= c.detect[0] && d < c.detect[1]
					if d < c.detect[0]-1000 {
						t.Errorf("false alarm at row %d", d)
					}
				}
				if !found {
					t.Errorf("drift not detected in rows %v: %v", c.detect, drifts)
				}
			}
		})
	}
}
