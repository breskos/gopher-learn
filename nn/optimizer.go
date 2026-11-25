package nn

import "math"

// Optimizer turns gradients into weight updates.
//
// An optimizer keeps state per parameter (e.g. momentum), so use one
// optimizer per network and do not share it between networks.
type Optimizer interface {
	// Step updates params in place from the mean gradients of a batch. key
	// identifies the parameter slice and is stable across calls.
	Step(key int, params, grads []float64, learningRate float64)
}

type sgd struct{}

// SGD returns plain stochastic gradient descent: w -= lr * g.
func SGD() Optimizer { return sgd{} }

func (sgd) Step(_ int, params, grads []float64, lr float64) {
	for i, g := range grads {
		params[i] -= lr * g
	}
}

type momentum struct {
	beta     float64
	velocity map[int][]float64
}

// Momentum returns gradient descent with momentum: the update keeps a share
// beta (typically 0.9) of the previous update, which smooths and speeds up
// training.
func Momentum(beta float64) Optimizer {
	return &momentum{beta: beta, velocity: map[int][]float64{}}
}

func (m *momentum) Step(key int, params, grads []float64, lr float64) {
	v := m.velocity[key]
	if len(v) != len(params) {
		v = make([]float64, len(params))
		m.velocity[key] = v
	}
	for i, g := range grads {
		v[i] = m.beta*v[i] - lr*g
		params[i] += v[i]
	}
}

type adam struct {
	beta1, beta2, eps float64
	m, v              map[int][]float64
	t                 map[int]int
}

// Adam returns the Adam optimizer with the given decay rates for the first
// and second moment estimates and a small epsilon for numerical stability.
// DefaultAdam uses the common values.
func Adam(beta1, beta2, eps float64) Optimizer {
	return &adam{beta1: beta1, beta2: beta2, eps: eps, m: map[int][]float64{}, v: map[int][]float64{}, t: map[int]int{}}
}

// DefaultAdam returns Adam with beta1 = 0.9, beta2 = 0.999 and eps = 1e-8.
// It works well with learning rates around 0.001 to 0.01.
func DefaultAdam() Optimizer { return Adam(0.9, 0.999, 1e-8) }

func (a *adam) Step(key int, params, grads []float64, lr float64) {
	m, v := a.m[key], a.v[key]
	if len(m) != len(params) {
		m, v = make([]float64, len(params)), make([]float64, len(params))
		a.m[key], a.v[key], a.t[key] = m, v, 0
	}
	a.t[key]++
	t := float64(a.t[key])
	c1 := 1 - math.Pow(a.beta1, t)
	c2 := 1 - math.Pow(a.beta2, t)
	for i, g := range grads {
		m[i] = a.beta1*m[i] + (1-a.beta1)*g
		v[i] = a.beta2*v[i] + (1-a.beta2)*g*g
		params[i] -= lr * (m[i] / c1) / (math.Sqrt(v[i]/c2) + a.eps)
	}
}
