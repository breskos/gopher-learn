package nn

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"
)

func TestFloat32MatchesFloat64(t *testing.T) {
	for i, act := range []Activation{Sigmoid, Tanh, ReLU, Linear} {
		n, _ := New(Config{Inputs: 30, Hidden: []int{20, 10}, Labels: []string{"a", "b", "c"}, HiddenActivation: act, OutputActivation: Softmax, LayerNorm: i%2 == 1, Rand: seeded()})
		c := n.Float32()
		if err := c.Validate(); err != nil {
			t.Fatal(err)
		}
		p := c.NewPredictor()
		rng := rand.New(rand.NewPCG(1, 1))
		for range 50 {
			x := make([]float64, 30)
			x32 := make([]float32, 30)
			for i := range x {
				x[i] = rng.NormFloat64()
				x32[i] = float32(x[i])
			}
			want, _ := n.Predict(x)
			got, err := p.Predict(x32)
			if err != nil {
				t.Fatal(err)
			}
			for i := range want {
				if math.Abs(float64(got[i])-want[i]) > 1e-5 {
					t.Fatalf("%v: output %d is %v in float32, %v in float64", act, i, got[i], want[i])
				}
			}
			wantLabel, _ := n.Classify(x)
			if l, _ := p.Classify(x32); l != wantLabel && math.Abs(want[0]-want[1]) > 1e-4 {
				t.Fatalf("%v: label %q, want %q", act, l, wantLabel)
			}
		}
		if allocs := testing.AllocsPerRun(100, func() { _, _ = p.Predict(make([]float32, 30)[:30]) }); allocs > 1 {
			t.Fatalf("Predict allocates %v times", allocs)
		}
	}
}

func TestFloat32Binary(t *testing.T) {
	n, _ := New(Config{Inputs: 5, Hidden: []int{4}, Labels: []string{"yes", "no"}, OutputActivation: Softmax, Rand: seeded()})
	c := n.Float32()
	normed, _ := New(Config{Inputs: 5, Hidden: []int{4}, Outputs: 1, LayerNorm: true, Rand: seeded()})
	nb, err := normed.Float32().MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	var nd Network32
	if err := nd.UnmarshalBinary(nb); err != nil || !nd.Layers[0].Norm || len(nd.Layers[0].Gain) != 4 {
		t.Fatalf("normalized network: %v", err)
	}
	b, err := c.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if want := 12 + 2*16 + 4*(20+4+8+2) + 4 + 3 + 4 + 2; len(b) != want {
		t.Fatalf("encoded %d bytes, want %d", len(b), want)
	}
	var d Network32
	if err := d.UnmarshalBinary(b); err != nil {
		t.Fatal(err)
	}
	x := []float32{1, 2, 3, 4, 5}
	a, _ := c.NewPredictor().Predict(x)
	got, _ := d.NewPredictor().Predict(x)
	if a[0] != got[0] || a[1] != got[1] || d.Labels[1] != "no" {
		t.Fatalf("round trip: %v vs %v, labels %v", a, got, d.Labels)
	}

	// every truncation and some corruptions fail cleanly
	for i := range len(b) {
		if err := new(Network32).UnmarshalBinary(b[:i]); err == nil {
			t.Fatalf("truncated to %d bytes: expected error", i)
		}
	}
	corrupt := func(at int, v byte) []byte {
		c := append([]byte(nil), b...)
		c[at] = v
		return c
	}
	for name, data := range map[string][]byte{
		"magic":      corrupt(0, 'X'),
		"no layers":  corrupt(4, 0),
		"huge":       corrupt(15, 0x7f),
		"activation": corrupt(20, 9),
		"flags":      corrupt(24, 4),
		"trailing":   append(append([]byte(nil), b...), 0),
	} {
		if err := new(Network32).UnmarshalBinary(data); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
	if _, err := (&Network32{}).MarshalBinary(); err == nil {
		t.Error("empty network: expected error")
	}
}

func TestFloat32Errors(t *testing.T) {
	n, _ := New(Config{Inputs: 2, Outputs: 1, Rand: seeded()})
	c := n.Float32()
	p := c.NewPredictor()
	if _, err := p.Predict([]float32{1}); !errors.Is(err, ErrDimension) {
		t.Fatalf("want ErrDimension, got %v", err)
	}
	if _, err := p.Classify([]float32{1, 2}); !errors.Is(err, ErrNoLabels) {
		t.Fatalf("want ErrNoLabels, got %v", err)
	}
	c.Layers = append(c.Layers, &Layer32{Inputs: 1, Outputs: 1, Weights: []float32{1}, Biases: []float32{0}})
	if _, err := p.Predict([]float32{1, 2}); !errors.Is(err, ErrDimension) {
		t.Fatalf("stale predictor: %v", err)
	}
	c.Layers[1].Outputs = 3
	p = c.NewPredictor()
	c.Layers[1].Outputs = 1
	if _, err := p.Predict([]float32{1, 2}); !errors.Is(err, ErrDimension) {
		t.Fatalf("changed layer: %v", err)
	}
	cases := map[string]func(*Network32){
		"nil layer":  func(c *Network32) { c.Layers[0] = nil },
		"shape":      func(c *Network32) { c.Layers[0].Inputs = 0 },
		"weights":    func(c *Network32) { c.Layers[0].Weights = nil },
		"activation": func(c *Network32) { c.Layers[0].Activation = Activation(9) },
		"softmax":    func(c *Network32) { c.Layers[0].Activation = Softmax },
		"chain":      func(c *Network32) { c.Layers[1].Inputs = 2; c.Layers[1].Weights = []float32{1, 1} },
		"labels":     func(c *Network32) { c.Labels = []string{"a", "b"} },
		"norm":       func(c *Network32) { c.Layers[0].Norm = true },
	}
	for name, mutate := range cases {
		c := n.Float32()
		c.Layers = append(c.Layers, &Layer32{Inputs: 1, Outputs: 1, Weights: []float32{1}, Biases: []float32{0}})
		mutate(c)
		if err := c.Validate(); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
	if err := (&Network32{}).Validate(); err == nil {
		t.Error("no layers: expected error")
	}
}

func FuzzNetwork32Binary(f *testing.F) {
	n, _ := New(Config{Inputs: 3, Hidden: []int{2}, Labels: []string{"a", "b"}, Rand: seeded()})
	b, _ := n.Float32().MarshalBinary()
	f.Add(b)
	f.Add([]byte("GLN1"))
	f.Fuzz(func(t *testing.T, data []byte) {
		var c Network32
		if err := c.UnmarshalBinary(data); err != nil {
			return
		}
		out, err := c.NewPredictor().Predict(make([]float32, c.InputSize()))
		if err != nil || len(out) != c.OutputSize() {
			t.Fatalf("decoded network does not predict: %v", err)
		}
	})
}
