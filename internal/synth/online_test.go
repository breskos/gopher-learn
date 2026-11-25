package synth_test

import (
	"context"
	"math"
	"sort"
	"testing"

	"github.com/breskos/gopher-learn/internal/synth"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/online"
)

// stream feeds the first `until` rows of a data set to an online learner in
// order, retraining every 250 rows, and returns the learner and all rows.
func stream(t *testing.T, path string, rows, until int, scale bool) (*online.Online, []*learn.Sample) {
	t.Helper()
	d, err := synth.ByPath(path)
	if err != nil {
		t.Fatal(err)
	}
	if !d.Ordered {
		t.Fatalf("%s is not marked as ordered", path)
	}
	data, err := d.Set(rows, 42)
	if err != nil {
		t.Fatal(err)
	}
	if scale {
		head := &learn.Set{Task: data.Task, Samples: data.Samples[:until]}
		sc, err := learn.FitScaler(head, learn.Standardize)
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
	cfg := online.DefaultConfig()
	cfg.Seed = 1
	cfg.RegressionThreshold = 0.05
	o, err := online.New(set, data.InputSize(), []int{12}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i, s := range data.Samples[:until] {
		c := *s
		if err := o.Inject(&c, true); err != nil {
			t.Fatal(err)
		}
		if (i+1)%250 == 0 {
			if _, err := o.Iterate(context.Background()); err != nil {
				t.Fatal(err)
			}
		}
	}
	return o, data.Samples
}

func accuracyOn(t *testing.T, n *nn.Network, samples []*learn.Sample) float64 {
	t.Helper()
	correct := 0
	for _, s := range samples {
		out, err := n.Predict(s.Vector)
		if err != nil {
			t.Fatal(err)
		}
		if nn.ArgMax(out) == s.Class {
			correct++
		}
	}
	return float64(correct) / float64(len(samples))
}

func mseOn(t *testing.T, n *nn.Network, samples []*learn.Sample) float64 {
	t.Helper()
	var sum float64
	for _, s := range samples {
		out, err := n.Predict(s.Vector)
		if err != nil {
			t.Fatal(err)
		}
		sum += (out[0] - s.Value) * (out[0] - s.Value)
	}
	return sum / float64(len(samples))
}

// TestOnlineStreamsDrift checks that every drifting classification stream
// is learnable online before the drift and that the drift is real: the
// model trained on the old concept fails on the new one.
func TestOnlineStreamsDrift(t *testing.T) {
	if testing.Short() {
		t.Skip("quality tests train real models")
	}
	cases := []struct {
		path                string
		rows, until         int
		before, after       [2]int // evaluation windows
		minBefore, maxAfter float64
	}{
		// drift at row 4000
		{"online/abrupt-drift.csv", 8000, 3000, [2]int{3000, 4000}, [2]int{4000, 5000}, 0.85, 0.65},
		// concept 2 takes over between rows 3200 and 4800
		{"online/gradual-drift.csv", 8000, 3000, [2]int{3000, 3200}, [2]int{4800, 5800}, 0.85, 0.65},
		// concepts switch every 2000 rows
		{"online/recurring-concepts.csv", 16000, 1500, [2]int{1500, 2000}, [2]int{2000, 3000}, 0.85, 0.65},
	}
	for _, c := range cases {
		t.Run(c.path, func(t *testing.T) {
			o, rows := stream(t, c.path, c.rows, c.until, false)
			before := accuracyOn(t, o.Network, rows[c.before[0]:c.before[1]])
			after := accuracyOn(t, o.Network, rows[c.after[0]:c.after[1]])
			t.Logf("accuracy before drift %.3f, after drift %.3f", before, after)
			if before < c.minBefore {
				t.Errorf("not learnable online: %.3f before the drift, want at least %.2f", before, c.minBefore)
			}
			if after > c.maxAfter {
				t.Errorf("no drift: %.3f after the drift, want at most %.2f", after, c.maxAfter)
			}
		})
	}
}

func TestOnlineIncrementalDrift(t *testing.T) {
	if testing.Short() {
		t.Skip("quality tests train real models")
	}
	o, rows := stream(t, "online/incremental-drift.csv", 10000, 2000, false)
	early, late := mseOn(t, o.Network, rows[2000:3000]), mseOn(t, o.Network, rows[9000:])
	t.Logf("MSE right after training %.4f, at the end of the stream %.4f", early, late)
	if early > 0.01 {
		t.Errorf("not learnable online: MSE %.4f", early)
	}
	if late < 4*early {
		t.Errorf("drift too weak: MSE grows only from %.4f to %.4f", early, late)
	}
}

func TestOnlineRoomOccupancy(t *testing.T) {
	if testing.Short() {
		t.Skip("quality tests train real models")
	}
	// four weeks online, then predict the following week
	week := 7 * 288
	o, rows := stream(t, "online/room-occupancy.csv", 5*week, 4*week, true)
	acc := accuracyOn(t, o.Network, rows[4*week:])
	t.Logf("accuracy in week five: %.3f", acc)
	if acc < 0.9 {
		t.Errorf("accuracy %.3f below 0.90", acc)
	}
}

// TestAirQualityAging checks the drift of the aging sensor in the data
// itself. Humidity changes with the seasons, so the same season is compared
// across years: the ratio of raw reading to reference is higher in the
// second winter (dust in the optics) and drops after the replacement.
func TestAirQualityAging(t *testing.T) {
	d, err := synth.ByPath("online/air-quality-aging.csv")
	if err != nil {
		t.Fatal(err)
	}
	data, err := d.Set(d.Rows, 1)
	if err != nil {
		t.Fatal(err)
	}
	ratio := func(from int) float64 {
		var r []float64
		for _, s := range data.Samples[from : from+2000] {
			if s.Value > 0.02 {
				r = append(r, s.Vector[1]/(s.Value*150)) // raw_pm25 / reference
			}
		}
		sort.Float64s(r)
		return r[len(r)/2]
	}
	const year = 365 * 24
	replaced := len(data.Samples) * 2 / 3
	winter1, winter2 := ratio(0), ratio(year)
	beforeYear, after := ratio(replaced-year), ratio(replaced)
	t.Logf("median raw/reference: winter %.2f -> %.2f a year later; same season before replacement %.2f, after %.2f",
		winter1, winter2, beforeYear, after)
	if math.IsNaN(winter1) || winter2 < winter1*1.2 {
		t.Errorf("aging not visible: %.2f -> %.2f", winter1, winter2)
	}
	if after > beforeYear*0.85 {
		t.Errorf("replacement not visible: %.2f a year before, %.2f after", beforeYear, after)
	}
}
