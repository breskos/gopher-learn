package optim

import (
	"errors"
	"math"
	"testing"
)

func TestRosenbrock(t *testing.T) {
	f := func(x, g []float64) float64 {
		a, b := 1-x[0], x[1]-x[0]*x[0]
		g[0] = -2*a - 400*x[0]*b
		g[1] = 200 * b
		return a*a + 100*b*b
	}
	x0 := []float64{-1.2, 1}
	r, err := LBFGS(f, x0, Config{})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Converged || math.Abs(r.X[0]-1) > 1e-5 || math.Abs(r.X[1]-1) > 1e-5 {
		t.Fatalf("result %+v", r)
	}
	if x0[0] != -1.2 {
		t.Fatal("x0 was modified")
	}
	t.Logf("%d iterations", r.Iterations)
}

func TestQuadratic(t *testing.T) {
	// f = sum_i c_i (x_i - i)^2 with very different curvatures
	c := []float64{1, 10, 100, 1000, 1e4}
	f := func(x, g []float64) float64 {
		var v float64
		for i := range x {
			d := x[i] - float64(i)
			v += c[i] * d * d
			g[i] = 2 * c[i] * d
		}
		return v
	}
	r, err := LBFGS(f, make([]float64, 5), Config{Memory: 3, GradTol: 1e-9})
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range r.X {
		if math.Abs(v-float64(i)) > 1e-8 {
			t.Fatalf("x = %v", r.X)
		}
	}
	// stops after MaxIter
	r, _ = LBFGS(f, make([]float64, 5), Config{MaxIter: 2})
	if r.Iterations > 2 || r.Converged {
		t.Fatalf("MaxIter: %+v", r)
	}
}

func TestNonFinite(t *testing.T) {
	_, err := LBFGS(func(x, g []float64) float64 { return math.NaN() }, []float64{1}, Config{})
	if !errors.Is(err, ErrNonFinite) {
		t.Fatalf("want ErrNonFinite, got %v", err)
	}
	// a function that becomes infinite away from the start stops cleanly
	r, err := LBFGS(func(x, g []float64) float64 {
		if x[0] < 0.5 {
			g[0] = -1
			return -x[0] * 0
		}
		g[0] = -1
		return math.Inf(-1)
	}, []float64{0}, Config{MaxIter: 5})
	if err != nil || r.X[0] >= 0.5 {
		t.Fatalf("%+v, %v", r, err)
	}
	// a linear function has no minimum; the search ends without crashing
	r, err = LBFGS(func(x, g []float64) float64 { g[0] = 1; return x[0] }, []float64{0}, Config{MaxIter: 20})
	if err != nil || r.X[0] >= 0 {
		t.Fatalf("linear: %+v, %v", r, err)
	}
}
