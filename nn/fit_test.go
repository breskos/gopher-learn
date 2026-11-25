package nn

import (
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func TestOptimizerSteps(t *testing.T) {
	p := []float64{1, 1}
	SGD().Step(0, p, []float64{0.5, -1}, 0.1)
	if p[0] != 0.95 || p[1] != 1.1 {
		t.Fatalf("SGD: %v", p)
	}

	m := Momentum(0.9)
	p = []float64{0}
	m.Step(0, p, []float64{1}, 0.1) // v = -0.1
	m.Step(0, p, []float64{1}, 0.1) // v = -0.19
	if math.Abs(p[0]+0.29) > 1e-12 {
		t.Fatalf("Momentum: %v, want -0.29", p)
	}

	// Adam's first step moves every parameter by about lr, regardless of the gradient's scale.
	a := DefaultAdam()
	p = []float64{0, 0}
	a.Step(0, p, []float64{1000, -0.001}, 0.01)
	if math.Abs(p[0]+0.01) > 1e-6 || math.Abs(p[1]-0.01) > 1e-4 {
		t.Fatalf("Adam: %v", p)
	}
	// separate keys keep separate state
	q := []float64{0}
	a.Step(1, q, []float64{1}, 0.01)
	if math.Abs(q[0]+0.01) > 1e-6 {
		t.Fatalf("Adam key 1: %v", q)
	}
}

func TestWeightDecay(t *testing.T) {
	n, _ := New(Config{Inputs: 1, Outputs: 1, OutputActivation: Linear, Rand: seeded()})
	n.Layers[0].Weights[0], n.Layers[0].Biases[0] = 2, 2
	// zero gradient: target equals output 2*0 + 2 = 2
	_, err := n.TrainBatchWith([][]float64{{0}}, [][]float64{{2}}, Step{Optimizer: SGD(), LearningRate: 0.1, WeightDecay: 0.5})
	if err != nil {
		t.Fatal(err)
	}
	if w, b := n.Layers[0].Weights[0], n.Layers[0].Biases[0]; math.Abs(w-1.9) > 1e-12 || b != 2 {
		t.Fatalf("weight %v (want 1.9), bias %v (want 2)", w, b)
	}
	if _, err := n.TrainBatchWith([][]float64{{0}}, [][]float64{{2}}, Step{}); err == nil {
		t.Fatal("expected error without optimizer")
	}
}

// spirals returns two interleaved spirals in 2D.
func spirals(rng *rand.Rand, n int) Data {
	var d Data
	for range n {
		c := rng.IntN(2)
		r := 0.1 + rng.Float64()
		angle := r*2*math.Pi + float64(c)*math.Pi
		x := []float64{r*math.Cos(angle) + rng.NormFloat64()*0.02, r*math.Sin(angle) + rng.NormFloat64()*0.02}
		y := []float64{0, 0}
		y[c] = 1
		d.X, d.Y = append(d.X, x), append(d.Y, y)
	}
	return d
}

func accuracy(n *Network, d Data) float64 {
	correct := 0
	for i, x := range d.X {
		out, _ := n.Predict(x)
		if ArgMax(out) == ArgMax(d.Y[i]) {
			correct++
		}
	}
	return float64(correct) / float64(d.Len())
}

func TestFitSpiralsWithAdam(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 3))
	train, val := spirals(rng, 1500), spirals(rng, 500)
	// tanh + softmax with SGD at a high learning rate gets stuck; Adam does not.
	n, _ := New(Config{Inputs: 2, Hidden: []int{32, 32}, Outputs: 2, HiddenActivation: Tanh, OutputActivation: Softmax, Rand: rng})
	h, err := Fit(context.Background(), n, train, val, FitConfig{Epochs: 200, LearningRate: 0.01, Patience: 20, Rand: rng})
	if err != nil {
		t.Fatal(err)
	}
	acc := accuracy(n, val)
	t.Logf("stopped after %d epochs (%s), best epoch %d, validation accuracy %.3f", len(h.Epochs), h.Stop, h.BestEpoch, acc)
	if acc < 0.95 {
		t.Fatalf("validation accuracy %.3f", acc)
	}
	if h.Epochs[0].TrainLoss <= h.Epochs[h.BestEpoch].TrainLoss {
		t.Fatal("training loss did not decrease")
	}
}

func TestFitEarlyStoppingRestoresBest(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 5))
	// few noisy samples and a big network overfit quickly
	train, val := spirals(rng, 40), spirals(rng, 200)
	for i := range train.Y {
		if rng.Float64() < 0.3 {
			train.Y[i][0], train.Y[i][1] = train.Y[i][1], train.Y[i][0]
		}
	}
	n, _ := New(Config{Inputs: 2, Hidden: []int{64, 64}, Outputs: 2, HiddenActivation: Tanh, OutputActivation: Softmax, Rand: rng})
	h, err := Fit(context.Background(), n, train, val, FitConfig{Epochs: 500, LearningRate: 0.02, BatchSize: 8, Patience: 10, Rand: rng})
	if err != nil {
		t.Fatal(err)
	}
	if h.Stop != StopPatience {
		t.Fatalf("stop = %q after %d epochs", h.Stop, len(h.Epochs))
	}
	if len(h.Epochs) != h.BestEpoch+11 {
		t.Fatalf("stopped at %d epochs, best epoch %d, patience 10", len(h.Epochs), h.BestEpoch)
	}
	got, _ := n.Loss(val.X, val.Y)
	if want := h.Epochs[h.BestEpoch].ValLoss; math.Abs(got-want) > 1e-12 {
		t.Fatalf("validation loss after Fit %.6f, best epoch had %.6f", got, want)
	}
}

func TestFitCallbackScheduleAndCancel(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	data := spirals(rng, 100)
	n, _ := New(Config{Inputs: 2, Hidden: []int{4}, Outputs: 2, Rand: rng})
	var rates []float64
	h, err := Fit(context.Background(), n, data, Data{}, FitConfig{
		Epochs: 10, LearningRate: 0.1, Schedule: StepDecay(2, 0.5), Optimizer: SGD(), Rand: rng,
		OnEpoch: func(s EpochStats) bool {
			rates = append(rates, s.LearningRate)
			return s.Epoch < 4
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if h.Stop != StopCallback || len(h.Epochs) != 5 {
		t.Fatalf("stop %q after %d epochs", h.Stop, len(h.Epochs))
	}
	if want := []float64{0.1, 0.1, 0.05, 0.05, 0.025}; !equalApprox(rates, want) {
		t.Fatalf("learning rates %v, want %v", rates, want)
	}
	if h.Epochs[0].HasVal {
		t.Fatal("no validation data given")
	}
	if c := CosineDecay(10); c(0, 1) != 1 || math.Abs(c(5, 1)-0.5) > 1e-12 || c(10, 1) > 1e-12 || c(20, 1) > 1e-12 {
		t.Fatal("cosine schedule")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Fit(ctx, n, data, Data{}, FitConfig{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
}

func TestFitDiverges(t *testing.T) {
	n, _ := New(Config{Inputs: 1, Hidden: []int{4}, Outputs: 1, HiddenActivation: Linear, OutputActivation: Linear, Rand: seeded()})
	data := Data{X: [][]float64{{1}, {2}, {3}}, Y: [][]float64{{10}, {20}, {30}}}
	h, err := Fit(context.Background(), n, data, Data{}, FitConfig{Epochs: 200, LearningRate: 10, Optimizer: SGD(), BatchSize: 1})
	if err != nil {
		t.Fatal(err)
	}
	if h.Stop != StopDiverged {
		t.Fatalf("stop = %q", h.Stop)
	}
}

func TestFitErrors(t *testing.T) {
	n, _ := New(Config{Inputs: 2, Outputs: 2, Rand: seeded()})
	ok := Data{X: [][]float64{{1, 2}}, Y: [][]float64{{1, 0}}}
	cases := map[string]struct {
		train, val Data
		cfg        FitConfig
	}{
		"no data":       {Data{}, Data{}, FitConfig{}},
		"lengths":       {Data{X: ok.X}, Data{}, FitConfig{}},
		"patience":      {ok, Data{}, FitConfig{Patience: 3}},
		"dimension":     {Data{X: [][]float64{{1}}, Y: [][]float64{{1, 0}}}, Data{}, FitConfig{}},
		"val dimension": {ok, Data{X: [][]float64{{1}}, Y: [][]float64{{1, 0}}}, FitConfig{}},
	}
	for name, c := range cases {
		if _, err := Fit(context.Background(), n, c.train, c.val, c.cfg); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
	if _, err := n.Loss([][]float64{{1, 2}}, nil); err == nil {
		t.Error("Loss: expected length error")
	}
	if l, err := n.Loss(nil, nil); err != nil || l != 0 {
		t.Errorf("Loss of no samples: %v, %v", l, err)
	}
	if _, err := n.Loss([][]float64{{1, 2}}, [][]float64{{1}}); !errors.Is(err, ErrDimension) {
		t.Errorf("Loss: want ErrDimension, got %v", err)
	}
}

func equalApprox(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-12 {
			return false
		}
	}
	return true
}

// TestParallelTrainingIsDeterministic checks that the number of workers
// does not change the result, down to the last bit.
func TestParallelTrainingIsDeterministic(t *testing.T) {
	old := parallelMinWork
	parallelMinWork = 0
	defer func() { parallelMinWork = old }()

	rng := rand.New(rand.NewPCG(8, 8))
	data := spirals(rng, 100)
	var results [][]float64
	for _, workers := range []int{1, 3, 8} {
		n, _ := New(Config{Inputs: 2, Hidden: []int{40, 30}, Outputs: 2, HiddenActivation: Tanh, OutputActivation: Softmax, Rand: rand.New(rand.NewPCG(1, 1))})
		if n.small() {
			t.Fatal("the test needs the batched path")
		}
		for range 20 {
			if _, err := n.TrainBatchWith(data.X, data.Y, Step{Optimizer: SGD(), LearningRate: 0.1, Workers: workers}); err != nil {
				t.Fatal(err)
			}
		}
		var w []float64
		for _, l := range n.Layers {
			w = append(w, l.Weights...)
			w = append(w, l.Biases...)
		}
		results = append(results, w)
	}
	for i := 1; i < len(results); i++ {
		for j := range results[0] {
			if results[i][j] != results[0][j] {
				t.Fatalf("weights differ with %d workers at %d", []int{1, 3, 8}[i], j)
			}
		}
	}
}

// TestBatchMatchesSingleSamples checks that a batch update equals the mean
// of the single-sample gradients.
func TestBatchMatchesSingleSamples(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 9))
	data := spirals(rng, 5)
	a, _ := New(Config{Inputs: 2, Hidden: []int{4}, Outputs: 2, HiddenActivation: Tanh, Rand: rand.New(rand.NewPCG(1, 1))})
	b := a.Clone()
	if _, err := a.TrainBatch(data.X, data.Y, 0.5); err != nil {
		t.Fatal(err)
	}
	// the same update from accumulated single-sample gradients
	sum := make([][]float64, len(b.Layers))
	for i, l := range b.Layers {
		sum[i] = make([]float64, len(l.Weights))
	}
	for k := range data.X {
		c := b.Clone()
		if _, err := c.TrainBatch(data.X[k:k+1], data.Y[k:k+1], 1); err != nil {
			t.Fatal(err)
		}
		for i, l := range c.Layers {
			for j, w := range l.Weights {
				sum[i][j] += b.Layers[i].Weights[j] - w // gradient of sample k
			}
		}
	}
	for i, l := range b.Layers {
		for j := range l.Weights {
			want := l.Weights[j] - 0.5*sum[i][j]/float64(len(data.X))
			if math.Abs(a.Layers[i].Weights[j]-want) > 1e-12 {
				t.Fatalf("layer %d weight %d: batch %v, single samples %v", i, j, a.Layers[i].Weights[j], want)
			}
		}
	}
}

// TestSmallMatchesBatched checks that the sample-by-sample path for small
// networks computes the same gradients as the batched path.
func TestSmallMatchesBatched(t *testing.T) {
	rng := rand.New(rand.NewPCG(10, 10))
	data := spirals(rng, 9)
	for _, act := range []Activation{Sigmoid, ReLU, Linear, Softmax} {
		n, _ := New(Config{Inputs: 2, Hidden: []int{6, 5}, Outputs: 2, HiddenActivation: Tanh, OutputActivation: act, Rand: rand.New(rand.NewPCG(2, 2))})
		if !n.small() {
			t.Fatal("the test needs a small network")
		}
		s := n.trainState(len(data.X))
		for k, x := range data.X {
			copy(s.acts[0][k*2:], x)
		}
		lossSmall := n.trainSmall(s, len(data.X), data.Y)
		gw := make([][]float64, len(n.Layers))
		for i := range n.Layers {
			gw[i] = append(slices.Clone(s.gradW[i]), s.gradB[i]...)
		}
		n.forwardBatch(s, len(data.X), 1)
		lossBatch := n.outputDeltas(s, len(data.X), data.Y)
		n.backwardBatch(s, len(data.X), 1)
		if math.Abs(lossSmall-lossBatch) > 1e-12 {
			t.Fatalf("%v: loss %v vs %v", act, lossSmall, lossBatch)
		}
		for i := range n.Layers {
			for j, g := range append(slices.Clone(s.gradW[i]), s.gradB[i]...) {
				if math.Abs(g-gw[i][j]) > 1e-12 {
					t.Fatalf("%v: layer %d gradient %d: %v vs %v", act, i, j, gw[i][j], g)
				}
			}
		}
	}
}
