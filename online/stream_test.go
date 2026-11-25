package online

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

func classifier(t *testing.T, cfg Config) *Online {
	t.Helper()
	set := learn.NewSet(nn.Classification)
	_ = set.AddClass("A")
	_ = set.AddClass("B")
	cfg.Seed = 3
	o, err := New(set, 2, []int{6}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return o
}

// point returns a sample whose label follows the concept: x > 0 is A, or
// the opposite if flipped.
func point(rng *rand.Rand, flipped bool) *learn.Sample {
	x, y := rng.NormFloat64(), rng.NormFloat64()
	label := "A"
	if (x > 0) == flipped {
		label = "B"
	}
	return learn.NewClassificationSample([]float64{x, y}, label)
}

func TestLearnWindowAndDrift(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MemorySize = 1000
	var events []DriftEvent
	cfg.OnDrift = func(e DriftEvent) { events = append(events, e) }
	o := classifier(t, cfg)
	rng := rand.New(rand.NewPCG(1, 1))
	var drifts []int
	for i := range 4000 {
		if i == 2300 {
			// the memory must hold mainly the new concept by now
			newer := 0
			for _, s := range o.Data.Samples {
				if s.Vector[0] > 0 == (s.Label == "B") {
					newer++
				}
			}
			if share := float64(newer) / float64(o.Data.Len()); share < 0.8 {
				t.Fatalf("%.2f of %d stored samples follow the new concept", share, o.Data.Len())
			}
		}
		r, err := o.Learn(point(rng, i >= 2000))
		if err != nil {
			t.Fatal(err)
		}
		if r.Label != "A" && r.Label != "B" || len(r.Output) != 2 {
			t.Fatalf("result %+v", r)
		}
		if r.Drift {
			drifts = append(drifts, i)
		}
		if (i+1)%250 == 0 && o.Data.Len() >= cfg.MinimumDataPoints {
			if _, err := o.Iterate(context.Background()); err != nil {
				t.Fatal(err)
			}
		}
		if o.Data.Len() > 1000 {
			t.Fatalf("memory holds %d samples", o.Data.Len())
		}
	}
	if len(drifts) == 0 || drifts[0] < 2000 || drifts[0] > 2300 {
		t.Fatalf("drifts at %v", drifts)
	}
	if len(events) != len(drifts) || o.Drifts != len(drifts) || events[0].Sample != drifts[0]+1 {
		t.Fatalf("events %+v for drifts %v", events, drifts)
	}
	if events[0].Kept+events[0].Forgotten != 1000 {
		t.Fatalf("event %+v", events[0])
	}
	for i := 1; i < len(drifts); i++ {
		if drifts[i]-drifts[i-1] <= cfg.DriftCooldown {
			t.Fatalf("drifts %v closer than the cooldown", drifts)
		}
	}
	if o.Data.Len() != 1000 {
		t.Fatalf("memory refilled to %d", o.Data.Len())
	}
	if o.Learned != 4000 || o.Prequential.Cumulative.Correct+o.Prequential.Cumulative.Wrong != 4000 {
		t.Fatal("counters")
	}
}

// With softmax outputs the network adapts to the flipped concept within a
// few samples (sigmoid outputs trained on squared error saturate and stay
// stuck, which is why softmax is recommended for online classification).
func TestLearnSoftmaxRecovers(t *testing.T) {
	cfg := DefaultConfig()
	cfg.HiddenActivation, cfg.OutputActivation = nn.Tanh, nn.Softmax
	o := classifier(t, cfg)
	rng := rand.New(rand.NewPCG(1, 1))
	for i := range 4000 {
		if _, err := o.Learn(point(rng, i >= 2000)); err != nil {
			t.Fatal(err)
		}
	}
	if acc := o.Prequential.Accuracy(); acc < 0.95 {
		t.Fatalf("prequential accuracy %.3f after the change", acc)
	}
}

func TestLearnReservoir(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Memory, cfg.MemorySize, cfg.Drift = Reservoir, 200, NoDrift
	o := classifier(t, cfg)
	rng := rand.New(rand.NewPCG(2, 2))
	samples := map[*learn.Sample]int{}
	for i := range 4000 {
		s := point(rng, false)
		samples[s] = i
		if _, err := o.Learn(s); err != nil {
			t.Fatal(err)
		}
	}
	if o.Data.Len() != 200 || o.Seen != 4000 {
		t.Fatalf("%d stored, %d seen", o.Data.Len(), o.Seen)
	}
	// a uniform sample of the stream: about half from the first half
	early := 0
	for _, s := range o.Data.Samples {
		if samples[s] < 2000 {
			early++
		}
	}
	if early < 70 || early > 130 {
		t.Fatalf("%d of 200 samples from the first half", early)
	}

	// a drift clears the reservoir
	cfg.Drift = PageHinkley
	cfg.DriftDelta = 20
	o = classifier(t, cfg)
	var drift bool
	for i := range 3000 {
		r, err := o.Learn(point(rng, i >= 1500))
		if err != nil {
			t.Fatal(err)
		}
		if r.Drift && !drift {
			drift = true
			if o.Data.Len() != 1 || o.Seen != 1 {
				t.Fatalf("reservoir not cleared: %d samples, seen %d", o.Data.Len(), o.Seen)
			}
		}
	}
	if !drift || o.DriftPageHinkley == nil || o.DriftPageHinkley.Threshold != 20 {
		t.Fatal("Page-Hinkley did not detect the drift")
	}
}

func TestLearnRegressionUnbounded(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Memory, cfg.Drift = Unbounded, PageHinkley
	cfg.OutputActivation = nn.Linear
	o, err := New(learn.NewSet(nn.Regression), 1, []int{4}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	// files written before v0.6 have no prequential window; Learn uses 500
	o.Config.PrequentialWindow = 0
	for i := range 6000 {
		x := float64(i%10) / 10
		if _, err := o.Learn(learn.NewRegressionSample([]float64{x}, 0.5*x)); err != nil {
			t.Fatal(err)
		}
	}
	if o.Data.Len() != 6000 || o.Prequential.Window != 500 {
		t.Fatalf("%d samples, window %d", o.Data.Len(), o.Prequential.Window)
	}
	if mae := o.Prequential.MAE(); mae > 0.05 {
		t.Fatalf("MAE %.3f", mae)
	}
}

func TestLearnErrors(t *testing.T) {
	o := classifier(t, DefaultConfig())
	if _, err := o.Learn(nil); err == nil {
		t.Error("nil sample: expected error")
	}
	if _, err := o.Learn(learn.NewClassificationSample([]float64{1}, "A")); !errors.Is(err, nn.ErrDimension) {
		t.Errorf("dimension: %v", err)
	}
	if _, err := o.Learn(learn.NewClassificationSample([]float64{1, 2}, "C")); !errors.Is(err, learn.ErrUnknownClass) {
		t.Errorf("unknown class: %v", err)
	}
	if _, err := o.Learn(learn.NewClassificationSample([]float64{1, math.Inf(1)}, "A")); !errors.Is(err, learn.ErrNonFinite) {
		t.Errorf("non-finite: %v", err)
	}
	if o.Learned != 0 || o.Prequential != nil {
		t.Fatal("rejected samples must not be evaluated")
	}

	// a full reservoir reports errors of the replaced sample
	o.Config.Memory, o.Config.MemorySize = Reservoir, 1
	if err := o.Inject(learn.NewClassificationSample([]float64{1, 2}, "A"), false); err != nil {
		t.Fatal(err)
	}
	o.Seen = 0
	if err := o.Inject(learn.NewClassificationSample([]float64{3, 4}, "C"), false); !errors.Is(err, learn.ErrUnknownClass) {
		t.Errorf("reservoir replace: %v", err)
	}
	if _, err := o.Learn(&learn.Sample{Vector: []float64{1, 2}, Label: "A", Value: math.NaN()}); !errors.Is(err, learn.ErrNonFinite) {
		t.Errorf("non-finite target: %v", err)
	}
}

func TestStreamConfig(t *testing.T) {
	cases := map[string]func(*Config){
		"memory":      func(c *Config) { c.Memory = Memory(5) },
		"memory size": func(c *Config) { c.MemorySize = 3 },
		"detector":    func(c *Config) { c.Drift = Detector(-1) },
		"delta":       func(c *Config) { c.DriftDelta = -1 },
		"cooldown":    func(c *Config) { c.DriftCooldown = -1 },
		"window":      func(c *Config) { c.PrequentialWindow = 0 },
	}
	for name, mutate := range cases {
		cfg := DefaultConfig()
		mutate(&cfg)
		if err := cfg.Validate(); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
	cfg := DefaultConfig()
	cfg.Memory, cfg.MemorySize = Unbounded, 0
	if err := cfg.Validate(); err != nil {
		t.Errorf("unbounded memory needs no size: %v", err)
	}

	b, err := json.Marshal(DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	var got Config
	if err := json.Unmarshal(b, &got); err != nil || got.Memory != Window || got.Drift != ADWIN {
		t.Fatalf("round trip: %+v, %v", got, err)
	}
	for _, s := range []string{`{"memory":"nope"}`, `{"drift":"nope"}`} {
		if err := json.Unmarshal([]byte(s), &got); err == nil {
			t.Errorf("%s: expected error", s)
		}
	}
	if _, err := json.Marshal(Config{Memory: Memory(9)}); err == nil {
		t.Error("invalid memory: expected error")
	}
	if _, err := json.Marshal(Config{Drift: Detector(9)}); err == nil {
		t.Error("invalid detector: expected error")
	}
	if Memory(9).String() != "Memory(9)" || Reservoir.String() != "reservoir" || PageHinkley.String() != "page-hinkley" || Detector(9).String() != "Detector(9)" {
		t.Error("names")
	}
}
