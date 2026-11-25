package nn

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// Network32 is an inference-only copy of a Network with float32 weights. It
// needs half the memory and predicts faster on small devices. Create it
// from a trained network with Network.Float32; it cannot be trained.
//
// Network32 has a compact binary encoding (MarshalBinary) that loads
// without reflection or a JSON parser, which suits embedded targets.
type Network32 struct {
	Layers []*Layer32
	Labels []string
}

// Layer32 is a dense layer with float32 weights.
type Layer32 struct {
	Inputs, Outputs int
	Weights         []float32 // Outputs × Inputs, row-major
	Biases          []float32
	Activation      Activation
	// Norm enables layer normalization with Gain and Shift.
	Norm        bool
	Gain, Shift []float32
}

// Float32 returns an inference-only float32 copy of the network.
func (n *Network) Float32() *Network32 {
	c := &Network32{Layers: make([]*Layer32, len(n.Layers)), Labels: append([]string(nil), n.Labels...)}
	for i, l := range n.Layers {
		c.Layers[i] = &Layer32{
			Inputs: l.Inputs, Outputs: l.Outputs, Activation: l.Activation,
			Weights: toFloat32(l.Weights), Biases: toFloat32(l.Biases),
		}
		if l.Norm {
			c.Layers[i].Norm, c.Layers[i].Gain, c.Layers[i].Shift = true, toFloat32(l.Gain), toFloat32(l.Shift)
		}
	}
	return c
}

func toFloat32(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}

// Validate checks the structural consistency of the network.
func (n *Network32) Validate() error {
	if len(n.Layers) == 0 {
		return errors.New("nn: network has no layers")
	}
	for i, l := range n.Layers {
		switch {
		case l == nil:
			return fmt.Errorf("nn: layer %d is nil", i)
		case l.Inputs <= 0 || l.Outputs <= 0:
			return fmt.Errorf("nn: layer %d has invalid shape %dx%d", i, l.Outputs, l.Inputs)
		case len(l.Weights) != l.Inputs*l.Outputs || len(l.Biases) != l.Outputs:
			return fmt.Errorf("nn: layer %d has %d weights and %d biases for shape %dx%d", i, len(l.Weights), len(l.Biases), l.Outputs, l.Inputs)
		case !l.Activation.valid():
			return fmt.Errorf("nn: layer %d has invalid activation", i)
		case l.Activation == Softmax && i != len(n.Layers)-1:
			return fmt.Errorf("nn: softmax is only allowed on the output layer (layer %d)", i)
		case i > 0 && l.Inputs != n.Layers[i-1].Outputs:
			return fmt.Errorf("nn: layer %d expects %d inputs, previous layer has %d outputs", i, l.Inputs, n.Layers[i-1].Outputs)
		case l.Norm && (len(l.Gain) != l.Outputs || len(l.Shift) != l.Outputs):
			return fmt.Errorf("nn: layer %d has invalid layer normalization", i)
		}
	}
	if len(n.Labels) > 0 && len(n.Labels) != n.OutputSize() {
		return fmt.Errorf("nn: %d labels for %d outputs", len(n.Labels), n.OutputSize())
	}
	return nil
}

// InputSize returns the expected length of input vectors.
func (n *Network32) InputSize() int { return n.Layers[0].Inputs }

// OutputSize returns the length of output vectors.
func (n *Network32) OutputSize() int { return n.Layers[len(n.Layers)-1].Outputs }

// Predictor32 runs forward passes of a Network32 without allocating. It must
// not be used by several goroutines at the same time.
type Predictor32 struct {
	n    *Network32
	acts [][]float32
}

// NewPredictor returns a Predictor32 for the network.
func (n *Network32) NewPredictor() *Predictor32 {
	p := &Predictor32{n: n, acts: make([][]float32, len(n.Layers))}
	for i, l := range n.Layers {
		p.acts[i] = make([]float32, l.Outputs)
	}
	return p
}

// Predict runs a forward pass. The returned slice belongs to the predictor
// and is overwritten by the next call.
func (p *Predictor32) Predict(x []float32) ([]float32, error) {
	n := p.n
	if len(x) != n.InputSize() {
		return nil, fmt.Errorf("%w: input has %d values, network expects %d", ErrDimension, len(x), n.InputSize())
	}
	if len(p.acts) != len(n.Layers) {
		return nil, fmt.Errorf("%w: the network changed its shape, create a new Predictor", ErrDimension)
	}
	in := x
	for i, l := range n.Layers {
		out := p.acts[i]
		if len(out) != l.Outputs {
			return nil, fmt.Errorf("%w: the network changed its shape, create a new Predictor", ErrDimension)
		}
		l.forward(in, out)
		in = out
	}
	return in, nil
}

// Classify returns the label of the output neuron with the highest activation.
func (p *Predictor32) Classify(x []float32) (string, error) {
	if len(p.n.Labels) == 0 {
		return "", ErrNoLabels
	}
	out, err := p.Predict(x)
	if err != nil {
		return "", err
	}
	best := 0
	for i, v := range out {
		if v > out[best] {
			best = i
		}
	}
	return p.n.Labels[best], nil
}

func (l *Layer32) forward(in, out []float32) {
	in = in[:l.Inputs]
	for o := range out {
		row := l.Weights[o*l.Inputs : (o+1)*l.Inputs]
		out[o] = l.Biases[o] + dot32(row, in)
	}
	if l.Norm {
		var mean, variance float64
		for _, v := range out {
			mean += float64(v)
		}
		mean /= float64(len(out))
		for _, v := range out {
			variance += (float64(v) - mean) * (float64(v) - mean)
		}
		istd := 1 / math.Sqrt(variance/float64(len(out))+normEpsilon)
		for j, v := range out {
			out[j] = l.Gain[j]*float32((float64(v)-mean)*istd) + l.Shift[j]
		}
	}
	l.Activation.apply32(out)
}

func dot32(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i+8 <= len(a); i += 8 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
		s4 += a[i+4] * b[i+4]
		s5 += a[i+5] * b[i+5]
		s6 += a[i+6] * b[i+6]
		s7 += a[i+7] * b[i+7]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
}

// apply32 computes the activation in place. Exponentials use float64
// internally for accuracy.
func (a Activation) apply32(v []float32) {
	switch a {
	case Sigmoid:
		for i, x := range v {
			v[i] = float32(1 / (1 + math.Exp(-float64(x))))
		}
	case Tanh:
		for i, x := range v {
			v[i] = float32(math.Tanh(float64(x)))
		}
	case ReLU:
		for i, x := range v {
			v[i] = max(x, 0)
		}
	case Softmax:
		m := float32(math.Inf(-1))
		for _, x := range v {
			m = max(m, x)
		}
		var sum float64
		for i, x := range v {
			e := math.Exp(float64(x - m))
			v[i] = float32(e)
			sum += e
		}
		for i := range v {
			v[i] = float32(float64(v[i]) / sum)
		}
	}
}

// binaryMagic starts the binary encoding of a Network32.
var binaryMagic = [4]byte{'G', 'L', 'N', '1'}

// flagNorm marks a layer with layer normalization in the binary encoding.
const flagNorm = 1

// Limits for decoding untrusted binary data.
var (
	maxBinaryLayers = 1 << 10
	maxBinaryValues = 1 << 28
	maxLabelLength  = 1 << 16
)

// MarshalBinary encodes the network in a compact little-endian format:
// magic "GLN1", the number of layers and labels, every layer (inputs,
// outputs, activation, flags, weights, biases and, with normalization,
// gains and shifts) and the labels.
func (n *Network32) MarshalBinary() ([]byte, error) {
	if err := n.Validate(); err != nil {
		return nil, err
	}
	size := 12
	for _, l := range n.Layers {
		size += 16 + 4*(len(l.Weights)+len(l.Biases)+len(l.Gain)+len(l.Shift))
	}
	for _, s := range n.Labels {
		size += 4 + len(s)
	}
	b := make([]byte, 0, size)
	b = append(b, binaryMagic[:]...)
	b = binary.LittleEndian.AppendUint32(b, uint32(len(n.Layers)))
	b = binary.LittleEndian.AppendUint32(b, uint32(len(n.Labels)))
	for _, l := range n.Layers {
		b = binary.LittleEndian.AppendUint32(b, uint32(l.Inputs))
		b = binary.LittleEndian.AppendUint32(b, uint32(l.Outputs))
		b = binary.LittleEndian.AppendUint32(b, uint32(l.Activation))
		var flags uint32
		if l.Norm {
			flags |= flagNorm
		}
		b = binary.LittleEndian.AppendUint32(b, flags)
		for _, v := range [][]float32{l.Weights, l.Biases, l.Gain, l.Shift} {
			for _, w := range v {
				b = binary.LittleEndian.AppendUint32(b, math.Float32bits(w))
			}
		}
	}
	for _, s := range n.Labels {
		b = binary.LittleEndian.AppendUint32(b, uint32(len(s)))
		b = append(b, s...)
	}
	return b, nil
}

// UnmarshalBinary decodes a network written by MarshalBinary. It checks
// every size against the remaining data before allocating, so corrupt or
// malicious input fails with an error instead of exhausting memory.
func (n *Network32) UnmarshalBinary(data []byte) error {
	r := reader{b: data}
	var magic [4]byte
	copy(magic[:], r.next(4))
	if r.err != nil || magic != binaryMagic {
		return errors.New("nn: not a binary float32 network")
	}
	layers, labels := r.uint(), r.uint()
	if r.err != nil || layers == 0 || layers > maxBinaryLayers || labels > maxBinaryValues {
		return errors.New("nn: invalid binary network header")
	}
	out := Network32{Layers: make([]*Layer32, 0, min(layers, 64))}
	total := 0
	for range layers {
		in, o, act, flags := r.uint(), r.uint(), r.uint(), r.uint()
		if r.err != nil || in == 0 || o == 0 || in > maxBinaryValues/o || flags&^flagNorm != 0 {
			return errors.New("nn: invalid binary layer")
		}
		values := in*o + o
		if flags&flagNorm != 0 {
			values += 2 * o
		}
		total += values
		if total > maxBinaryValues || r.left() < 4*values {
			return errors.New("nn: binary network is truncated or too large")
		}
		l := &Layer32{Inputs: in, Outputs: o, Activation: Activation(act), Weights: r.floats(in * o), Biases: r.floats(o)}
		if flags&flagNorm != 0 {
			l.Norm, l.Gain, l.Shift = true, r.floats(o), r.floats(o)
		}
		out.Layers = append(out.Layers, l)
	}
	if labels > r.left()/4 {
		return errors.New("nn: binary network is truncated")
	}
	for range labels {
		size := r.uint()
		if r.err != nil || size > maxLabelLength || size > r.left() {
			return errors.New("nn: invalid binary label")
		}
		out.Labels = append(out.Labels, string(r.next(size)))
	}
	if r.err != nil {
		return r.err
	}
	if r.left() != 0 {
		return errors.New("nn: trailing data after binary network")
	}
	if err := out.Validate(); err != nil {
		return err
	}
	*n = out
	return nil
}

// reader decodes little-endian values and remembers the first error.
type reader struct {
	b   []byte
	err error
}

func (r *reader) left() int { return len(r.b) }

func (r *reader) next(n int) []byte {
	if r.err != nil || n > len(r.b) {
		r.err = errors.New("nn: binary network is truncated")
		return nil
	}
	v := r.b[:n]
	r.b = r.b[n:]
	return v
}

// uint reads a size. Sizes above maxBinaryValues are errors, so they fit
// into an int on 32-bit platforms too.
func (r *reader) uint() int {
	b := r.next(4)
	if b == nil {
		return 0
	}
	v := binary.LittleEndian.Uint32(b)
	if v > uint32(maxBinaryValues) {
		r.err = errors.New("nn: size in binary network out of range")
		return 0
	}
	return int(v)
}

func (r *reader) floats(n int) []float32 {
	b := r.next(4 * n)
	if b == nil {
		return nil
	}
	v := make([]float32, n)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}
