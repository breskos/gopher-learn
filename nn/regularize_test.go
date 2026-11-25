package nn

import (
	"context"
	"encoding/json"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// TestLayerNormGradient compares the analytic gradients of a network with
// layer normalization against numerical ones, for weights, biases, gains
// and shifts.
func TestLayerNormGradient(t *testing.T) {
	for _, out := range []Activation{Linear, Softmax} {
		n, err := New(Config{Inputs: 3, Hidden: []int{5, 4}, Outputs: 2, HiddenActivation: Tanh, OutputActivation: out, LayerNorm: true, Rand: seeded()})
		if err != nil {
			t.Fatal(err)
		}
		// non-trivial gains and shifts
		rng := rand.New(rand.NewPCG(4, 4))
		for _, l := range n.Layers[:2] {
			for j := range l.Gain {
				l.Gain[j], l.Shift[j] = 0.5+rng.Float64(), rng.NormFloat64()*0.3
			}
		}
		xs := [][]float64{{0.3, -0.7, 0.2}, {1, 0.5, -1}}
		ys := [][]float64{{1, 0}, {0, 1}}
		loss := func() float64 {
			l, _ := n.Loss(xs, ys)
			return l
		}
		s := n.trainState(2)
		copy(s.acts[0], xs[0])
		copy(s.acts[0][3:], xs[1])
		n.forwardBatch(s, 2, 1, nil)
		n.outputDeltas(s, 2, ys)
		n.backwardBatch(s, 2, 1, false)
		check := func(name string, params, grads []float64) {
			for j := range params {
				orig := params[j]
				params[j] = orig + 1e-6
				plus := loss()
				params[j] = orig - 1e-6
				minus := loss()
				params[j] = orig
				num := (plus - minus) / 2e-6
				if got := grads[j] / 2; math.Abs(got-num) > 1e-6 { // grads are summed over 2 samples
					t.Fatalf("%v %s %d: analytic %g, numeric %g", out, name, j, got, num)
				}
			}
		}
		for i, l := range n.Layers {
			check("weight", l.Weights, s.gradW[i])
			check("bias", l.Biases, s.gradB[i])
			if l.Norm {
				check("gain", l.Gain, s.gradGain[i])
				check("shift", l.Shift, s.gradShift[i])
			}
		}
	}
}

func TestDropout(t *testing.T) {
	n, err := New(Config{Inputs: 50, Hidden: []int{200}, Outputs: 2, HiddenActivation: Sigmoid, OutputActivation: Softmax, Dropout: 0.3, Rand: seeded()})
	if err != nil {
		t.Fatal(err)
	}
	if n.Layers[0].Dropout != 0.3 || n.Layers[1].Dropout != 0 {
		t.Fatal("dropout belongs to the hidden layers only")
	}
	x := make([]float64, 50)
	s := n.trainState(1)
	copy(s.acts[0], x)
	n.forwardBatch(s, 1, 1, rand.New(rand.NewPCG(1, 1)).Float64)
	dropped := 0
	for _, m := range s.masks[0][:200] {
		switch m {
		case 0:
			dropped++
		case 1 / 0.7:
		default:
			t.Fatalf("mask value %v", m)
		}
	}
	if dropped < 40 || dropped > 80 {
		t.Fatalf("%d of 200 outputs dropped, want about 60", dropped)
	}
	// predictions use every neuron and are deterministic
	a, _ := n.Predict(x)
	b, _ := n.Predict(x)
	if !slices.Equal(a, b) {
		t.Fatal("prediction is not deterministic")
	}
}

func TestDropoutGradient(t *testing.T) {
	// with a fixed mask the gradient matches the numerical one of the
	// network with that mask
	n, _ := New(Config{Inputs: 3, Hidden: []int{30}, Outputs: 2, HiddenActivation: Tanh, OutputActivation: Softmax, Dropout: 0.5, Rand: seeded()})
	x, y := []float64{0.3, -0.7, 0.2}, []float64{1, 0}
	s := n.trainState(1)
	copy(s.acts[0], x)
	r := rand.New(rand.NewPCG(7, 7))
	n.forwardBatch(s, 1, 1, r.Float64)
	mask := slices.Clone(s.masks[0][:30])
	n.outputDeltas(s, 1, [][]float64{y})
	n.backwardBatch(s, 1, 1, true)
	masked := func() float64 {
		c := n.Clone()
		c.Layers[0].Dropout = 0
		for o, m := range mask { // fold the mask into the next layer
			for j := range c.Layers[1].Outputs {
				c.Layers[1].Weights[j*30+o] *= m
			}
		}
		l, _ := c.Loss([][]float64{x}, [][]float64{y})
		return l
	}
	w := n.Layers[0].Weights
	for j := range w {
		orig := w[j]
		w[j] = orig + 1e-6
		plus := masked()
		w[j] = orig - 1e-6
		minus := masked()
		w[j] = orig
		if num := (plus - minus) / 2e-6; math.Abs(num-s.gradW[0][j]) > 1e-6 {
			t.Fatalf("weight %d: analytic %g, numeric %g", j, s.gradW[0][j], num)
		}
	}
}

func TestRegularizedTraining(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 3))
	train, val := spirals(rng, 1500), spirals(rng, 500)
	n, _ := New(Config{Inputs: 2, Hidden: []int{48, 48}, Outputs: 2, HiddenActivation: Tanh, OutputActivation: Softmax, Dropout: 0.05, LayerNorm: true, Init: GlorotNormal, Rand: rng})
	h, err := Fit(context.Background(), n, train, val, FitConfig{Epochs: 150, LearningRate: 0.01, Patience: 20, Rand: rng})
	if err != nil {
		t.Fatal(err)
	}
	acc := accuracy(n, val)
	t.Logf("validation accuracy %.3f after %d epochs", acc, len(h.Epochs))
	if acc < 0.95 {
		t.Fatalf("accuracy %.3f", acc)
	}
	b, err := json.Marshal(n)
	if err != nil {
		t.Fatal(err)
	}
	var back Network
	if err := json.Unmarshal(b, &back); err != nil || back.Validate() != nil {
		t.Fatalf("round trip: %v", err)
	}
	if acc2 := accuracy(&back, val); acc2 != acc {
		t.Fatalf("accuracy after round trip %.3f", acc2)
	}
}

func TestInit(t *testing.T) {
	for _, c := range []struct {
		init     Init
		act      Activation
		variance float64
	}{
		{InitAuto, Tanh, 2.0 / 600},
		{InitAuto, ReLU, 2.0 / 400},
		{GlorotUniform, ReLU, 2.0 / 600},
		{GlorotNormal, Tanh, 2.0 / 600},
		{HeUniform, Tanh, 2.0 / 400},
		{HeNormal, Tanh, 2.0 / 400},
	} {
		n, err := New(Config{Inputs: 400, Outputs: 200, OutputActivation: c.act, Init: c.init, Rand: seeded()})
		if err != nil {
			t.Fatal(err)
		}
		var ss float64
		for _, w := range n.Layers[0].Weights {
			ss += w * w
		}
		if v := ss / float64(len(n.Layers[0].Weights)); math.Abs(v-c.variance)/c.variance > 0.05 {
			t.Errorf("init %d with %v: variance %.5f, want %.5f", c.init, c.act, v, c.variance)
		}
	}
	for name, cfg := range map[string]Config{
		"init":    {Inputs: 1, Outputs: 1, Init: Init(9)},
		"dropout": {Inputs: 1, Outputs: 1, Dropout: 1},
		"nan":     {Inputs: 1, Outputs: 1, Dropout: math.NaN()},
	} {
		if _, err := New(cfg); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
}

func TestValidateRegularization(t *testing.T) {
	cases := map[string]func(n *Network){
		"output dropout":  func(n *Network) { n.Layers[1].Dropout = 0.1 },
		"dropout range":   func(n *Network) { n.Layers[0].Dropout = 1 },
		"output norm":     func(n *Network) { n.Layers[1].Norm = true },
		"gain length":     func(n *Network) { n.Layers[0].Gain = n.Layers[0].Gain[:1] },
		"params w/o norm": func(n *Network) { n.Layers[0].Norm = false },
	}
	for name, mutate := range cases {
		n, _ := New(Config{Inputs: 2, Hidden: []int{3}, Outputs: 1, LayerNorm: true, Rand: seeded()})
		mutate(n)
		if err := n.Validate(); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
}
