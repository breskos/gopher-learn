package linear

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

// exact returns samples of y = 3 x1 - 2 x2 + 0.005 x3 + 7 with x3 on a
// scale of thousands.
func exact(n int, noise float64, seed uint64) *learn.Set {
	rng := rand.New(rand.NewPCG(seed, seed))
	s := learn.NewSet(nn.Regression)
	s.Features = []string{"a", "b", "c"}
	for range n {
		x := []float64{rng.NormFloat64(), rng.Float64(), rng.Float64() * 5000}
		y := 3*x[0] - 2*x[1] + 0.005*x[2] + 7 + noise*rng.NormFloat64()
		_ = s.Add(learn.NewRegressionSample(x, y))
	}
	return s
}

func TestRegressionExact(t *testing.T) {
	for _, n := range []int{50, 20000} { // the second one takes the parallel path
		r, err := FitRegression(exact(n, 0, 1), RegressionConfig{})
		if err != nil {
			t.Fatal(err)
		}
		want := []float64{3, -2, 0.005}
		for j, w := range r.Weights {
			if math.Abs(w-want[j]) > 1e-9*math.Max(1, math.Abs(want[j])) {
				t.Fatalf("n=%d: weights %v", n, r.Weights)
			}
		}
		if math.Abs(r.Bias-7) > 1e-7 || r.Features[2] != "c" {
			t.Fatalf("n=%d: bias %v, features %v", n, r.Bias, r.Features)
		}
		if err := r.Validate(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestRegressionNoiseAndRidge(t *testing.T) {
	train, test := exact(2000, 0.5, 2), exact(500, 0.5, 3)
	r, err := FitRegression(train, RegressionConfig{})
	if err != nil {
		t.Fatal(err)
	}
	e, err := r.Evaluate(test, 1)
	if err != nil {
		t.Fatal(err)
	}
	if e.R2() < 0.95 || math.Abs(e.RMSE()-0.5) > 0.05 {
		t.Fatalf("R² %.3f, RMSE %.3f (noise 0.5)", e.R2(), e.RMSE())
	}
	ridge, _ := FitRegression(train, RegressionConfig{L2: 10})
	norm := func(w []float64) float64 { return math.Abs(w[0]) + math.Abs(w[1]) }
	if norm(ridge.Weights) >= norm(r.Weights)/2 {
		t.Fatalf("ridge does not shrink: %v vs %v", ridge.Weights, r.Weights)
	}
}

func TestRegressionSingular(t *testing.T) {
	// the second feature duplicates the first: no unique solution
	s := learn.NewSet(nn.Regression)
	rng := rand.New(rand.NewPCG(4, 4))
	for range 100 {
		x := rng.NormFloat64()
		_ = s.Add(learn.NewRegressionSample([]float64{x, x, 1}, 2*x+1))
	}
	r, err := FitRegression(s, RegressionConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if y, _ := r.Predict([]float64{1.5, 1.5, 1}); math.Abs(y-4) > 1e-6 {
		t.Fatalf("prediction %v, weights %v", y, r.Weights)
	}
	// the system is singular and only stabilized by a tiny ridge, so the
	// split between the duplicates is exact only up to rounding (which
	// differs between CPUs); their sum is what matters
	if math.Abs(r.Weights[0]+r.Weights[1]-2) > 1e-9 || math.Abs(r.Weights[0]-r.Weights[1]) > 1e-4 {
		t.Fatalf("duplicates should share the weight: %v", r.Weights)
	}
}

// blobs returns k Gaussian clusters in d dimensions.
func blobs(n, k, d int, spread float64, seed uint64) *learn.Set {
	rng := rand.New(rand.NewPCG(seed, seed))
	centers := make([][]float64, k)
	for c := range centers {
		centers[c] = make([]float64, d)
		for j := range centers[c] {
			centers[c][j] = rng.Float64()*10 - 5
		}
	}
	labels := []string{"a", "b", "c", "d", "e", "f"}[:k]
	s := learn.NewSet(nn.Classification)
	for _, l := range labels {
		_ = s.AddClass(l)
	}
	for i := range n {
		c := i % k
		x := make([]float64, d)
		for j := range x {
			x[j] = centers[c][j] + spread*rng.NormFloat64()
		}
		_ = s.Add(learn.NewClassificationSample(x, labels[c]))
	}
	return s
}

func TestLogisticMulticlass(t *testing.T) {
	train, test := blobs(4000, 4, 8, 1.5, 5), blobs(1000, 4, 8, 1.5, 5)
	m, err := FitLogistic(train, LogisticConfig{})
	if err != nil {
		t.Fatal(err)
	}
	e, err := m.Evaluate(test)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("accuracy %.3f, log loss %.3f, %d iterations, converged %v", e.CorrectRatio(), e.LogLoss(), m.Iterations, m.Converged)
	if !m.Converged || e.CorrectRatio() < 0.97 {
		t.Fatalf("accuracy %.3f, converged %v", e.CorrectRatio(), m.Converged)
	}
	p, _ := m.Probabilities(test.Samples[0].Vector)
	var sum float64
	for _, v := range p {
		sum += v
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Fatalf("probabilities sum to %v", sum)
	}
	if l, _ := m.Classify(test.Samples[0].Vector); l != test.Samples[0].Label {
		t.Fatalf("classified %q, want %q", l, test.Samples[0].Label)
	}
	if err := m.Validate(); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(m)
	var back Logistic
	if err := json.Unmarshal(b, &back); err != nil || back.Validate() != nil {
		t.Fatal("JSON round trip")
	}
}

func TestLogisticCoefficients(t *testing.T) {
	// P(yes) rises with x1 (in thousands), falls with x2 and ignores x3
	rng := rand.New(rand.NewPCG(6, 6))
	s := learn.NewSet(nn.Classification)
	_ = s.AddClass("no")
	_ = s.AddClass("yes")
	for range 5000 {
		x := []float64{rng.Float64() * 4000, rng.NormFloat64(), rng.NormFloat64()}
		logit := 0.002*(x[0]-2000) - 1.5*x[1]
		label := "no"
		if rng.Float64() < 1/(1+math.Exp(-logit)) {
			label = "yes"
		}
		_ = s.Add(learn.NewClassificationSample(x, label))
	}
	m, err := FitLogistic(s, LogisticConfig{L2: -1})
	if err != nil {
		t.Fatal(err)
	}
	// with two classes, the difference of the rows is the classic logit
	d := func(j int) float64 { return m.Weights[1][j] - m.Weights[0][j] }
	t.Logf("logit weights %.4f %.3f %.3f", d(0), d(1), d(2))
	if math.Abs(d(0)-0.002) > 0.0004 || math.Abs(d(1)+1.5) > 0.2 || math.Abs(d(2)) > 0.15 {
		t.Fatalf("logit weights %.4f %.3f %.3f, want 0.002 -1.5 0", d(0), d(1), d(2))
	}
}

func TestErrors(t *testing.T) {
	reg, cls := exact(10, 0, 1), blobs(20, 2, 2, 1, 1)
	if _, err := FitRegression(nil, RegressionConfig{}); err == nil {
		t.Error("nil set")
	}
	if _, err := FitRegression(cls, RegressionConfig{}); err == nil {
		t.Error("classification set for regression")
	}
	if _, err := FitRegression(reg, RegressionConfig{L2: -1}); err == nil {
		t.Error("negative L2")
	}
	if _, err := FitLogistic(reg, LogisticConfig{}); err == nil {
		t.Error("regression set for logistic")
	}
	if _, err := FitLogistic(cls, LogisticConfig{L2: math.NaN()}); err == nil {
		t.Error("NaN L2")
	}
	one := learn.NewSet(nn.Classification)
	_ = one.AddClass("a")
	_ = one.Add(learn.NewClassificationSample([]float64{1}, "a"))
	_ = one.Add(learn.NewClassificationSample([]float64{2}, "a"))
	if _, err := FitLogistic(one, LogisticConfig{}); err == nil {
		t.Error("one class")
	}
	bad := exact(10, 0, 1)
	bad.Samples[0].Vector[0] = math.Inf(1)
	if _, err := FitRegression(bad, RegressionConfig{}); err == nil {
		t.Error("infinite value")
	}

	r, _ := FitRegression(reg, RegressionConfig{})
	if _, err := r.Predict([]float64{1}); !errors.Is(err, nn.ErrDimension) {
		t.Error("Predict dimension")
	}
	if _, err := r.Evaluate(blobs(4, 2, 1, 1, 1), 0); err == nil {
		t.Error("Evaluate dimension")
	}
	for name, mutate := range map[string]func(*Regression){
		"empty":    func(r *Regression) { r.Weights = nil },
		"features": func(r *Regression) { r.Features = []string{"a"} },
		"nan":      func(r *Regression) { r.Bias = math.NaN() },
	} {
		c := *r
		mutate(&c)
		if c.Validate() == nil {
			t.Errorf("Regression.Validate %s", name)
		}
	}

	m, _ := FitLogistic(cls, LogisticConfig{MaxIter: 3})
	if m.Iterations > 3 {
		t.Errorf("MaxIter: %d iterations", m.Iterations)
	}
	if _, err := m.Probabilities([]float64{1}); !errors.Is(err, nn.ErrDimension) {
		t.Error("Probabilities dimension")
	}
	if err := m.ProbabilitiesInto(make([]float64, 5), []float64{1, 2}); !errors.Is(err, nn.ErrDimension) {
		t.Error("ProbabilitiesInto dst")
	}
	if _, err := m.Classify([]float64{1}); err == nil {
		t.Error("Classify dimension")
	}
	if _, err := m.Evaluate(blobs(10, 3, 2, 1, 1)); err == nil {
		t.Error("Evaluate with other labels")
	}
	other := blobs(10, 2, 3, 1, 1)
	if _, err := m.Evaluate(other); err == nil {
		t.Error("Evaluate dimension")
	}
	for name, mutate := range map[string]func(*Logistic){
		"labels":   func(m *Logistic) { m.Labels = m.Labels[:1] },
		"empty":    func(m *Logistic) { m.Weights = [][]float64{{}, {}} },
		"ragged":   func(m *Logistic) { m.Weights = [][]float64{{1, 2}, {1}} },
		"features": func(m *Logistic) { m.Features = []string{"x"} },
		"nan":      func(m *Logistic) { m.Biases = []float64{0, math.NaN()} },
	} {
		c := *m
		mutate(&c)
		if c.Validate() == nil {
			t.Errorf("Logistic.Validate %s", name)
		}
	}
}
