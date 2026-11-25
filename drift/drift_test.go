package drift

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"testing"
)

func bernoulli(rng *rand.Rand, p float64) float64 {
	if rng.Float64() < p {
		return 1
	}
	return 0
}

// stream feeds a detector and returns the indices where it signaled drift.
func stream(d Detector, n int, value func(i int) float64) []int {
	var hits []int
	for i := range n {
		if d.Add(value(i)) {
			hits = append(hits, i)
		}
	}
	return hits
}

func TestADWINStationary(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	a := NewADWIN(0.002)
	hits := stream(a, 20000, func(int) float64 { return bernoulli(rng, 0.2) })
	if len(hits) > 1 {
		t.Fatalf("%d false alarms at %v", len(hits), hits)
	}
	if a.Width < 10000 || math.Abs(a.Mean()-0.2) > 0.02 {
		t.Fatalf("width %d, mean %.3f", a.Width, a.Mean())
	}
	// memory grows logarithmically
	buckets := 0
	for _, l := range a.Levels {
		buckets += len(l)
	}
	if buckets > 6*int(math.Log2(float64(a.Width))) {
		t.Fatalf("%d buckets for %d values", buckets, a.Width)
	}
}

func TestADWINAbrupt(t *testing.T) {
	for _, c := range []struct {
		name   string
		before float64
		after  float64
		delay  int
	}{
		{"error rate rises", 0.2, 0.6, 150},
		{"error rate falls", 0.5, 0.1, 150},
		{"small change", 0.2, 0.35, 600},
	} {
		rng := rand.New(rand.NewPCG(2, 2))
		a := NewADWIN(0.002)
		hits := stream(a, 6000, func(i int) float64 {
			if i < 3000 {
				return bernoulli(rng, c.before)
			}
			return bernoulli(rng, c.after)
		})
		if len(hits) == 0 || hits[0] < 3000 || hits[0] > 3000+c.delay {
			t.Errorf("%s: detections at %v, want the first within %d after 3000", c.name, hits, c.delay)
			continue
		}
		if math.Abs(a.Mean()-c.after) > 0.05 || a.Width > 3000+c.delay {
			t.Errorf("%s: window of %d values with mean %.3f after the change", c.name, a.Width, a.Mean())
		}
		if (a.Change > 0) != (c.after > c.before) {
			t.Errorf("%s: change %.3f has the wrong sign", c.name, a.Change)
		}
		t.Logf("%s: detected after %d values", c.name, hits[0]-3000)
	}
}

func TestADWINExactStatistics(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 3))
	a := &ADWIN{Delta: 1e-9, MaxBuckets: 2, MinWindow: 5, Clock: 7}
	var xs []float64
	for range 1000 {
		x := rng.Float64()
		xs = append(xs, x)
		a.Add(x)
	}
	var sum, ss float64
	for _, x := range xs {
		sum += x
	}
	mean := sum / float64(len(xs))
	for _, x := range xs {
		ss += (x - mean) * (x - mean)
	}
	if a.Width != 1000 || math.Abs(a.Total-sum) > 1e-9 || math.Abs(a.Variance-ss) > 1e-6 {
		t.Fatalf("width %d total %.6f/%.6f variance %.6f/%.6f", a.Width, a.Total, sum, a.Variance, ss)
	}
	count := 0
	for i, l := range a.Levels {
		if len(l) > a.MaxBuckets {
			t.Fatalf("level %d has %d buckets", i, len(l))
		}
		for _, b := range l {
			if b.Count != 1<<i {
				t.Fatalf("level %d bucket of %d values", i, b.Count)
			}
			count += b.Count
		}
	}
	if count != a.Width {
		t.Fatalf("buckets hold %d values, width %d", count, a.Width)
	}
}

func TestADWINDropKeepsStatistics(t *testing.T) {
	a := NewADWIN(0.01)
	for i := range 2000 {
		x := 0.0
		if i >= 1000 {
			x = 1
		}
		a.Add(x)
	}
	// the window is now (almost) only ones; its statistics must match
	count := 0
	var total float64
	for _, l := range a.Levels {
		for _, b := range l {
			count += b.Count
			total += b.Total
		}
	}
	if count != a.Width || math.Abs(total-a.Total) > 1e-9 || a.Variance < 0 {
		t.Fatalf("width %d/%d, total %.3f/%.3f, variance %.3f", count, a.Width, total, a.Total, a.Variance)
	}
	if a.Mean() < 0.95 {
		t.Fatalf("mean %.3f after the change", a.Mean())
	}

	// dropping everything resets the statistics
	b := &ADWIN{Levels: [][]Bucket{{{Count: 1, Total: 1}}}, Width: 1, Total: 1}
	b.dropOldest()
	if b.Width != 0 || b.Total != 0 || len(b.Levels) != 0 {
		t.Fatalf("not empty: %+v", b)
	}
}

func TestADWINResetAndJSON(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 4))
	a := NewADWIN(0)
	if a.Delta != 0.002 {
		t.Fatalf("default delta %v", a.Delta)
	}
	if a.Mean() != 0 {
		t.Fatal("mean of an empty window")
	}
	for range 500 {
		a.Add(bernoulli(rng, 0.3))
	}
	b, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	var c ADWIN
	if err := json.Unmarshal(b, &c); err != nil {
		t.Fatal(err)
	}
	for i := range 3000 {
		x := bernoulli(rng, 0.3)
		if i > 1000 {
			x = bernoulli(rng, 0.9)
		}
		if a.Add(x) != c.Add(x) {
			t.Fatalf("restored detector differs at %d", i)
		}
	}
	a.Reset()
	if a.Width != 0 || a.Levels != nil || a.Delta != 0.002 {
		t.Fatalf("reset: %+v", a)
	}
}

func TestADWINClock(t *testing.T) {
	a := &ADWIN{Clock: 32}
	hits := stream(a, 4000, func(i int) float64 {
		if i < 2000 {
			return 0
		}
		return 1
	})
	if len(hits) == 0 || hits[0]%32 != 31 {
		t.Fatalf("detections %v must fall on clock ticks", hits)
	}
}

func TestPageHinkley(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 5))
	p := NewPageHinkley()
	if hits := stream(p, 20000, func(int) float64 { return bernoulli(rng, 0.2) }); len(hits) > 0 {
		t.Fatalf("false alarms at %v", hits)
	}
	p.Reset()
	hits := stream(p, 6000, func(i int) float64 {
		if i < 3000 {
			return bernoulli(rng, 0.2)
		}
		return bernoulli(rng, 0.6)
	})
	if len(hits) == 0 || hits[0] < 3000 || hits[0] > 3300 {
		t.Fatalf("detections %v, want the first within 300 after 3000", hits)
	}
	t.Logf("detected after %d values", hits[0]-3000)
	if p.Count >= 3000 {
		t.Fatal("the test must start over after a detection")
	}

	// a falling error rate is not drift for Page-Hinkley
	p = &PageHinkley{Threshold: 20}
	if hits := stream(p, 6000, func(i int) float64 {
		if i < 3000 {
			return bernoulli(rng, 0.6)
		}
		return bernoulli(rng, 0.1)
	}); len(hits) > 0 {
		t.Fatalf("decrease detected at %v", hits)
	}
	if p.Threshold != 20 || p.Alpha != 0.9999 || p.MinSamples != 30 || p.Delta != 0.005 {
		t.Fatalf("defaults: %+v", p)
	}
}

func FuzzADWIN(f *testing.F) {
	f.Add(0.5, 0.1, uint64(1))
	f.Fuzz(func(t *testing.T, a, b float64, seed uint64) {
		if math.IsNaN(a) || math.IsNaN(b) || math.Abs(a) > 1e6 || math.Abs(b) > 1e6 {
			return
		}
		rng := rand.New(rand.NewPCG(seed, seed))
		d := NewADWIN(0.002)
		for i := range 300 {
			x := a
			if rng.IntN(2) == 0 || i > 150 {
				x = b
			}
			d.Add(x)
			if d.Width <= 0 || d.Variance < 0 || math.IsNaN(d.Variance) {
				t.Fatalf("inconsistent window: width %d variance %v", d.Width, d.Variance)
			}
		}
	})
}

var _ Detector = (*ADWIN)(nil)
var _ Detector = (*PageHinkley)(nil)
