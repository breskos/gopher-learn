// Package optim minimizes smooth functions for the classic models.
package optim

import (
	"errors"
	"math"
)

// Func returns f(x) and writes the gradient into grad.
type Func func(x, grad []float64) float64

// Config configures LBFGS. The zero value uses the defaults.
type Config struct {
	// MaxIter bounds the number of iterations. Default 500.
	MaxIter int
	// Memory is the number of correction pairs kept. Default 10.
	Memory int
	// GradTol stops when the largest gradient entry falls below it. Default 1e-6.
	GradTol float64
}

// Result describes the outcome of LBFGS.
type Result struct {
	X          []float64
	F          float64
	Iterations int
	Converged  bool
}

// ErrNonFinite is returned when the function value or gradient is NaN or
// infinite.
var ErrNonFinite = errors.New("optim: non-finite function value or gradient")

// LBFGS minimizes f starting at x0 with the limited-memory BFGS method and
// a backtracking line search (Armijo condition). x0 is not modified.
func LBFGS(f Func, x0 []float64, cfg Config) (Result, error) {
	if cfg.MaxIter <= 0 {
		cfg.MaxIter = 500
	}
	if cfg.Memory <= 0 {
		cfg.Memory = 10
	}
	if cfg.GradTol <= 0 {
		cfg.GradTol = 1e-6
	}
	n := len(x0)
	x := append([]float64(nil), x0...)
	g := make([]float64, n)
	fx := f(x, g)
	if !finite(fx) || !allFinite(g) {
		return Result{X: x, F: fx}, ErrNonFinite
	}
	var (
		s, y  [][]float64
		rho   []float64
		d     = make([]float64, n)
		xNew  = make([]float64, n)
		gNew  = make([]float64, n)
		alpha = make([]float64, cfg.Memory)
	)
	res := Result{X: x, F: fx}
	for it := 0; it < cfg.MaxIter; it++ {
		if maxAbs(g) < cfg.GradTol {
			res.Converged = true
			break
		}
		// two-loop recursion: d = -H·g
		copy(d, g)
		for i := len(s) - 1; i >= 0; i-- {
			alpha[i] = rho[i] * dot(s[i], d)
			axpy(-alpha[i], y[i], d)
		}
		// initial scaling of the inverse Hessian
		gamma := 1 / math.Max(1, math.Sqrt(dot(g, g)))
		if k := len(s) - 1; k >= 0 {
			gamma = dot(s[k], y[k]) / dot(y[k], y[k])
		}
		for i := range d {
			d[i] *= gamma
		}
		for i := range s {
			beta := rho[i] * dot(y[i], d)
			axpy(alpha[i]-beta, s[i], d)
		}
		for i := range d {
			d[i] = -d[i]
		}
		slope := dot(g, d)
		if slope >= 0 { // not a descent direction: restart with the gradient
			s, y, rho = s[:0], y[:0], rho[:0]
			for i := range d {
				d[i] = -g[i]
			}
			slope = -dot(g, g)
		}
		// backtracking line search
		step, fNew := 1.0, 0.0
		accepted := false
		for range 40 {
			for i := range x {
				xNew[i] = x[i] + step*d[i]
			}
			fNew = f(xNew, gNew)
			if finite(fNew) && fNew <= fx+1e-4*step*slope {
				accepted = true
				break
			}
			step *= 0.5
		}
		if !accepted || !allFinite(gNew) {
			break // no progress possible; x is the best point found
		}
		si := make([]float64, n)
		yi := make([]float64, n)
		for i := range x {
			si[i] = xNew[i] - x[i]
			yi[i] = gNew[i] - g[i]
		}
		if sy := dot(si, yi); sy > 1e-12 {
			if len(s) == cfg.Memory {
				s, y, rho = s[1:], y[1:], rho[1:]
			}
			s, y, rho = append(s, si), append(y, yi), append(rho, 1/sy)
		}
		copy(x, xNew)
		copy(g, gNew)
		improvement := fx - fNew
		fx = fNew
		res.Iterations = it + 1
		if improvement <= 1e-15*math.Abs(fx) {
			res.Converged = maxAbs(g) < math.Sqrt(cfg.GradTol)
			break
		}
	}
	res.X, res.F = x, fx
	return res, nil
}

func dot(a, b []float64) float64 {
	var s float64
	for i, v := range a {
		s += v * b[i]
	}
	return s
}

func axpy(a float64, x, y []float64) {
	for i, v := range x {
		y[i] += a * v
	}
}

func maxAbs(v []float64) float64 {
	var m float64
	for _, x := range v {
		m = max(m, math.Abs(x))
	}
	return m
}

func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

func allFinite(v []float64) bool {
	for _, x := range v {
		if !finite(x) {
			return false
		}
	}
	return true
}
