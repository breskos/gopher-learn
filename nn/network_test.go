package nn

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand/v2"
	"testing"
)

func seeded() *rand.Rand { return rand.New(rand.NewPCG(1, 2)) }

var (
	xorX = [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	xorY = [][]float64{{1, 0}, {0, 1}, {0, 1}, {1, 0}}
)

func trainXOR(t *testing.T, cfg Config, lr float64, epochs int) *Network {
	t.Helper()
	n, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for range epochs {
		if _, err := n.TrainBatch(xorX, xorY, lr); err != nil {
			t.Fatal(err)
		}
	}
	return n
}

func TestXOR(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
		lr   float64
	}{
		{"sigmoid", Config{Inputs: 2, Hidden: []int{8}, Labels: []string{"even", "odd"}}, 2},
		{"tanh-softmax", Config{Inputs: 2, Hidden: []int{8}, Labels: []string{"even", "odd"}, HiddenActivation: Tanh, OutputActivation: Softmax}, 0.5},
		{"relu-softmax", Config{Inputs: 2, Hidden: []int{16, 16}, Labels: []string{"even", "odd"}, HiddenActivation: ReLU, OutputActivation: Softmax}, 0.1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.cfg.Rand = seeded()
			n := trainXOR(t, tt.cfg, tt.lr, 5000)
			for i, x := range xorX {
				got, err := n.Classify(x)
				if err != nil {
					t.Fatal(err)
				}
				want := n.Labels[ArgMax(xorY[i])]
				if got != want {
					t.Errorf("Classify(%v) = %q, want %q", x, got, want)
				}
			}
		})
	}
}

func TestGradientMatchesNumerical(t *testing.T) {
	for _, act := range []Activation{Sigmoid, Tanh, Linear, Softmax} {
		t.Run(act.String(), func(t *testing.T) {
			n, err := New(Config{Inputs: 3, Hidden: []int{4}, Outputs: 2, HiddenActivation: Tanh, OutputActivation: act, Rand: seeded()})
			if err != nil {
				t.Fatal(err)
			}
			x := []float64{0.3, -0.7, 0.2}
			y := []float64{1, 0}
			lossOf := func() float64 {
				c := n.Clone()
				l, _ := c.Train(x, y, 0)
				return l
			}
			s := n.trainState()
			for i := range n.Layers {
				clear(s.gradW[i])
			}
			n.backprop(s, x, y)
			const eps = 1e-6
			for li, l := range n.Layers {
				for wi := range l.Weights {
					orig := l.Weights[wi]
					l.Weights[wi] = orig + eps
					plus := lossOf()
					l.Weights[wi] = orig - eps
					minus := lossOf()
					l.Weights[wi] = orig
					num := (plus - minus) / (2 * eps)
					if got := s.gradW[li][wi]; math.Abs(got-num) > 1e-5 {
						t.Fatalf("layer %d weight %d: analytic %g, numeric %g", li, wi, got, num)
					}
				}
			}
		})
	}
}

func TestPredictDimensionError(t *testing.T) {
	n, err := New(Config{Inputs: 2, Outputs: 1, Rand: seeded()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := n.Predict([]float64{1}); !errors.Is(err, ErrDimension) {
		t.Fatalf("want ErrDimension, got %v", err)
	}
	if _, err := n.Classify([]float64{1, 2}); !errors.Is(err, ErrNoLabels) {
		t.Fatalf("want ErrNoLabels, got %v", err)
	}
}

func TestNewValidation(t *testing.T) {
	bad := []Config{
		{Inputs: 0, Outputs: 1},
		{Inputs: 2},
		{Inputs: 2, Outputs: 3, Labels: []string{"a"}},
		{Inputs: 2, Outputs: 1, Hidden: []int{0}},
		{Inputs: 2, Outputs: 1, HiddenActivation: Softmax},
	}
	for i, cfg := range bad {
		if _, err := New(cfg); err == nil {
			t.Errorf("config %d: expected error", i)
		}
	}
}

func TestJSONRoundTrip(t *testing.T) {
	n := trainXOR(t, Config{Inputs: 2, Hidden: []int{4}, Labels: []string{"a", "b"}, OutputActivation: Softmax, Rand: seeded()}, 0.5, 10)
	b, err := json.Marshal(n)
	if err != nil {
		t.Fatal(err)
	}
	var m Network
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	if err := m.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, x := range xorX {
		a, _ := n.Predict(x)
		b, _ := m.Predict(x)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("prediction differs after round trip: %v vs %v", a, b)
			}
		}
	}
}

func TestCloneIsIndependent(t *testing.T) {
	n, _ := New(Config{Inputs: 2, Outputs: 2, Rand: seeded()})
	c := n.Clone()
	c.Layers[0].Weights[0] += 1
	if n.Layers[0].Weights[0] == c.Layers[0].Weights[0] {
		t.Fatal("clone shares weights with original")
	}
}

func BenchmarkTrain(b *testing.B) {
	n, _ := New(Config{Inputs: 60, Hidden: []int{100}, Outputs: 2, Rand: seeded()})
	x := make([]float64, 60)
	y := []float64{1, 0}
	for b.Loop() {
		_, _ = n.Train(x, y, 0.1)
	}
}

func BenchmarkPredict(b *testing.B) {
	n, _ := New(Config{Inputs: 60, Hidden: []int{100}, Outputs: 2, Rand: seeded()})
	x := make([]float64, 60)
	for b.Loop() {
		_, _ = n.Predict(x)
	}
}
