// Package nn implements a fully connected feed-forward neural network
// (multi-layer perceptron) backed by dense weight matrices.
//
// A Network is created with New, trained with Train or TrainBatch and used
// with Predict, Classify or Scores. Networks serialize to JSON directly.
//
// Predict, Classify and Scores are safe for concurrent use as long as no
// training happens at the same time. Train and TrainBatch are not safe for
// concurrent use.
package nn

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sync"
)

var (
	// ErrDimension is returned when a vector does not match the network shape.
	ErrDimension = errors.New("nn: dimension mismatch")
	// ErrNonFinite is returned by training when a loss or gradient is NaN or
	// infinite. The update is skipped, so the network stays usable.
	ErrNonFinite = errors.New("nn: non-finite loss or gradient")
	// ErrNoLabels is returned by Classify and Scores on a network without labels.
	ErrNoLabels = errors.New("nn: network has no labels")
)

// Config describes the shape of a new network.
type Config struct {
	// Inputs is the size of the input vector.
	Inputs int
	// Hidden holds the number of neurons of each hidden layer.
	Hidden []int
	// Outputs is the size of the output vector. If zero, len(Labels) is used.
	Outputs int
	// Labels optionally names the output neurons (used for classification).
	Labels []string
	// HiddenActivation is used for all hidden layers. Defaults to Sigmoid.
	HiddenActivation Activation
	// OutputActivation is used for the output layer. Defaults to Sigmoid.
	OutputActivation Activation
	// Rand is the source for weight initialization. If nil, a randomly
	// seeded source is used.
	Rand *rand.Rand
}

// Layer is a dense layer: out = activation(Weights · in + Biases).
type Layer struct {
	Inputs     int        `json:"inputs"`
	Outputs    int        `json:"outputs"`
	Weights    []float64  `json:"weights"` // Outputs × Inputs, row-major
	Biases     []float64  `json:"biases"`
	Activation Activation `json:"activation"`
}

// Network is a feed-forward neural network.
type Network struct {
	Layers []*Layer `json:"layers"`
	Labels []string `json:"labels,omitempty"`

	state *trainState
	pool  sync.Pool // *Predictor for Predict
}

// New creates a network with randomly initialized weights.
func New(cfg Config) (*Network, error) {
	outputs := cfg.Outputs
	if outputs == 0 {
		outputs = len(cfg.Labels)
	}
	if cfg.Inputs <= 0 {
		return nil, fmt.Errorf("nn: inputs must be positive, got %d", cfg.Inputs)
	}
	if outputs <= 0 {
		return nil, errors.New("nn: outputs must be positive (set Outputs or Labels)")
	}
	if len(cfg.Labels) > 0 && len(cfg.Labels) != outputs {
		return nil, fmt.Errorf("nn: %d labels for %d outputs", len(cfg.Labels), outputs)
	}
	if !cfg.HiddenActivation.valid() || cfg.HiddenActivation == Softmax {
		return nil, fmt.Errorf("nn: invalid hidden activation %v", cfg.HiddenActivation)
	}
	if !cfg.OutputActivation.valid() {
		return nil, fmt.Errorf("nn: invalid output activation %v", cfg.OutputActivation)
	}
	rng := cfg.Rand
	if rng == nil {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}

	sizes := append(append([]int{cfg.Inputs}, cfg.Hidden...), outputs)
	n := &Network{Labels: append([]string(nil), cfg.Labels...)}
	for i := 1; i < len(sizes); i++ {
		if sizes[i] <= 0 {
			return nil, fmt.Errorf("nn: hidden layer %d must be positive, got %d", i-1, sizes[i])
		}
		act := cfg.HiddenActivation
		if i == len(sizes)-1 {
			act = cfg.OutputActivation
		}
		n.Layers = append(n.Layers, newLayer(sizes[i-1], sizes[i], act, rng))
	}
	return n, nil
}

func newLayer(in, out int, act Activation, rng *rand.Rand) *Layer {
	l := &Layer{
		Inputs:     in,
		Outputs:    out,
		Weights:    make([]float64, in*out),
		Biases:     make([]float64, out),
		Activation: act,
	}
	// Glorot/Xavier uniform, He uniform for ReLU.
	limit := math.Sqrt(6 / float64(in+out))
	if act == ReLU {
		limit = math.Sqrt(6 / float64(in))
	}
	for i := range l.Weights {
		l.Weights[i] = (2*rng.Float64() - 1) * limit
	}
	return l
}

// Validate checks the structural consistency of the network,
// e.g. after decoding it from JSON.
func (n *Network) Validate() error {
	if len(n.Layers) == 0 {
		return errors.New("nn: network has no layers")
	}
	for i, l := range n.Layers {
		switch {
		case l == nil:
			return fmt.Errorf("nn: layer %d is nil", i)
		case l.Inputs <= 0 || l.Outputs <= 0:
			return fmt.Errorf("nn: layer %d has invalid shape %dx%d", i, l.Outputs, l.Inputs)
		case len(l.Weights) != l.Inputs*l.Outputs:
			return fmt.Errorf("nn: layer %d has %d weights, want %d", i, len(l.Weights), l.Inputs*l.Outputs)
		case len(l.Biases) != l.Outputs:
			return fmt.Errorf("nn: layer %d has %d biases, want %d", i, len(l.Biases), l.Outputs)
		case !l.Activation.valid():
			return fmt.Errorf("nn: layer %d has invalid activation", i)
		case l.Activation == Softmax && i != len(n.Layers)-1:
			return fmt.Errorf("nn: softmax is only allowed on the output layer (layer %d)", i)
		case i > 0 && l.Inputs != n.Layers[i-1].Outputs:
			return fmt.Errorf("nn: layer %d expects %d inputs, previous layer has %d outputs", i, l.Inputs, n.Layers[i-1].Outputs)
		}
	}
	if len(n.Labels) > 0 && len(n.Labels) != n.OutputSize() {
		return fmt.Errorf("nn: %d labels for %d outputs", len(n.Labels), n.OutputSize())
	}
	return nil
}

// InputSize returns the expected length of input vectors.
func (n *Network) InputSize() int { return n.Layers[0].Inputs }

// OutputSize returns the length of output vectors.
func (n *Network) OutputSize() int { return n.Layers[len(n.Layers)-1].Outputs }

// Predict runs a forward pass and returns a newly allocated output vector.
// It is safe for concurrent use; the intermediate activations come from a
// pool, so the output vector is the only allocation. Use a Predictor to
// avoid that one too.
func (n *Network) Predict(x []float64) ([]float64, error) {
	if len(x) != n.InputSize() {
		return nil, fmt.Errorf("%w: input has %d values, network expects %d", ErrDimension, len(x), n.InputSize())
	}
	p, _ := n.pool.Get().(*Predictor)
	if p == nil || !p.fits(n) {
		p = n.NewPredictor()
	}
	out, _ := p.Predict(x)
	out = slices.Clone(out)
	n.pool.Put(p)
	return out, nil
}

// Predictor runs forward passes of a network without allocating memory.
// It reuses its buffers, so it must not be used by several goroutines at
// the same time; create one per goroutine. If the shape of the network
// changes, create a new Predictor.
type Predictor struct {
	n    *Network
	acts [][]float64
}

// NewPredictor returns a Predictor for the network.
func (n *Network) NewPredictor() *Predictor {
	p := &Predictor{n: n, acts: make([][]float64, len(n.Layers))}
	for i, l := range n.Layers {
		p.acts[i] = make([]float64, l.Outputs)
	}
	return p
}

func (p *Predictor) fits(n *Network) bool {
	if p.n != n || len(p.acts) != len(n.Layers) {
		return false
	}
	for i, l := range n.Layers {
		if len(p.acts[i]) != l.Outputs {
			return false
		}
	}
	return true
}

// Predict runs a forward pass. The returned slice belongs to the Predictor
// and is overwritten by the next call; copy it to keep it.
func (p *Predictor) Predict(x []float64) ([]float64, error) {
	n := p.n
	if len(x) != n.InputSize() {
		return nil, fmt.Errorf("%w: input has %d values, network expects %d", ErrDimension, len(x), n.InputSize())
	}
	if !p.fits(n) {
		return nil, fmt.Errorf("%w: the network changed its shape, create a new Predictor", ErrDimension)
	}
	in := x
	for i, l := range n.Layers {
		l.forward(in, p.acts[i])
		in = p.acts[i]
	}
	return in, nil
}

// Classify returns the label of the output neuron with the highest
// activation without allocating.
func (p *Predictor) Classify(x []float64) (string, error) {
	if len(p.n.Labels) == 0 {
		return "", ErrNoLabels
	}
	out, err := p.Predict(x)
	if err != nil {
		return "", err
	}
	return p.n.Labels[ArgMax(out)], nil
}

// Classify returns the label of the output neuron with the highest activation.
func (n *Network) Classify(x []float64) (string, error) {
	if len(n.Labels) == 0 {
		return "", ErrNoLabels
	}
	if len(x) != n.InputSize() {
		return "", fmt.Errorf("%w: input has %d values, network expects %d", ErrDimension, len(x), n.InputSize())
	}
	p, _ := n.pool.Get().(*Predictor)
	if p == nil || !p.fits(n) {
		p = n.NewPredictor()
	}
	label, err := p.Classify(x)
	n.pool.Put(p)
	return label, err
}

// Scores returns the activation of every output neuron keyed by its label.
func (n *Network) Scores(x []float64) (map[string]float64, error) {
	if len(n.Labels) == 0 {
		return nil, ErrNoLabels
	}
	out, err := n.Predict(x)
	if err != nil {
		return nil, err
	}
	m := make(map[string]float64, len(out))
	for i, v := range out {
		m[n.Labels[i]] = v
	}
	return m, nil
}

// Clone returns a deep copy of the network.
func (n *Network) Clone() *Network {
	c := &Network{
		Layers: make([]*Layer, len(n.Layers)),
		Labels: append([]string(nil), n.Labels...),
	}
	for i, l := range n.Layers {
		lc := *l
		lc.Weights = append([]float64(nil), l.Weights...)
		lc.Biases = append([]float64(nil), l.Biases...)
		c.Layers[i] = &lc
	}
	return c
}

func (l *Layer) forward(in, out []float64) {
	in = in[:l.Inputs]
	out = out[:l.Outputs]
	for o := range out {
		row := l.Weights[o*l.Inputs : (o+1)*l.Inputs]
		if len(row) < 8 {
			sum := l.Biases[o]
			for j, w := range row {
				sum += w * in[j]
			}
			out[o] = sum
		} else {
			out[o] = l.Biases[o] + dot(row, in)
		}
	}
	l.Activation.apply(out, out)
}

// ArgMax returns the index of the largest value in v, or -1 if v is empty.
func ArgMax(v []float64) int {
	best := -1
	for i, x := range v {
		if best < 0 || x > v[best] {
			best = i
		}
	}
	return best
}
