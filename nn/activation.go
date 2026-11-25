package nn

import (
	"fmt"
	"math"
)

// Activation is the activation function of a layer.
type Activation int

const (
	// Sigmoid is the logistic function 1/(1+e^-x). It is the zero value.
	Sigmoid Activation = iota
	// Tanh is the hyperbolic tangent.
	Tanh
	// ReLU is max(0, x).
	ReLU
	// Linear is the identity function. Useful for unbounded regression outputs.
	Linear
	// Softmax normalizes the outputs to a probability distribution.
	// It is only valid on the output layer and trains with cross-entropy loss.
	Softmax
)

var activationNames = [...]string{"sigmoid", "tanh", "relu", "linear", "softmax"}

func (a Activation) valid() bool { return a >= 0 && int(a) < len(activationNames) }

// String returns the name of the activation.
func (a Activation) String() string {
	if !a.valid() {
		return fmt.Sprintf("Activation(%d)", int(a))
	}
	return activationNames[a]
}

// MarshalText implements encoding.TextMarshaler.
func (a Activation) MarshalText() ([]byte, error) {
	if !a.valid() {
		return nil, fmt.Errorf("nn: invalid activation %d", int(a))
	}
	return []byte(activationNames[a]), nil
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (a *Activation) UnmarshalText(b []byte) error {
	for i, name := range activationNames {
		if name == string(b) {
			*a = Activation(i)
			return nil
		}
	}
	return fmt.Errorf("nn: unknown activation %q", b)
}

// apply computes out = f(z). z and out may alias.
func (a Activation) apply(z, out []float64) {
	switch a {
	case Sigmoid:
		for i, v := range z {
			out[i] = 1 / (1 + math.Exp(-v))
		}
	case Tanh:
		for i, v := range z {
			out[i] = math.Tanh(v)
		}
	case ReLU:
		for i, v := range z {
			out[i] = max(v, 0)
		}
	case Linear:
		copy(out, z)
	case Softmax:
		m := math.Inf(-1)
		for _, v := range z {
			m = max(m, v)
		}
		var sum float64
		for i, v := range z {
			out[i] = math.Exp(v - m)
			sum += out[i]
		}
		for i := range out {
			out[i] /= sum
		}
	}
}

// derivative returns f'(z) expressed through the activation output y = f(z).
// Softmax is handled together with the cross-entropy loss and never reaches here.
func (a Activation) derivative(y float64) float64 {
	switch a {
	case Sigmoid:
		return y * (1 - y)
	case Tanh:
		return 1 - y*y
	case ReLU:
		if y > 0 {
			return 1
		}
		return 0
	default:
		return 1
	}
}
