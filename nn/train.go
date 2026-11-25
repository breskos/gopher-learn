package nn

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
)

// trainState holds reusable buffers for batched training. Activations and
// deltas of a batch are stored sample after sample in one slice per layer.
type trainState struct {
	batch  int         // capacity in samples
	acts   [][]float64 // acts[0] holds the inputs, acts[i+1] the outputs of layer i
	deltas [][]float64
	gradW  [][]float64
	gradB  [][]float64
}

func (n *Network) trainState(batch int) *trainState {
	s := n.state
	if s == nil || !s.fits(n) {
		s = &trainState{
			acts:   make([][]float64, len(n.Layers)+1),
			deltas: make([][]float64, len(n.Layers)),
			gradW:  make([][]float64, len(n.Layers)),
			gradB:  make([][]float64, len(n.Layers)),
		}
		for i, l := range n.Layers {
			s.gradW[i] = make([]float64, len(l.Weights))
			s.gradB[i] = make([]float64, l.Outputs)
		}
		n.state = s
	}
	if s.batch < batch {
		s.batch = batch
		s.acts[0] = make([]float64, batch*n.InputSize())
		for i, l := range n.Layers {
			s.acts[i+1] = make([]float64, batch*l.Outputs)
			s.deltas[i] = make([]float64, batch*l.Outputs)
		}
	}
	return s
}

// fits reports whether the buffers match the shape of the network.
func (s *trainState) fits(n *Network) bool {
	if len(s.gradW) != len(n.Layers) {
		return false
	}
	for i, l := range n.Layers {
		if len(s.gradW[i]) != len(l.Weights) || len(s.gradB[i]) != l.Outputs {
			return false
		}
	}
	return true
}

// parallelMinWork is the number of multiply-adds below which a layer is
// computed on one goroutine; starting goroutines costs more than it saves.
var parallelMinWork = 1 << 15

// split returns the number of goroutines for n items and the given work;
// 1 means the caller computes everything itself.
func split(n, workers, work int) int {
	if work < parallelMinWork {
		return 1
	}
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	return max(min(workers, n), 1)
}

// parallel calls fn on contiguous chunks of [0, n) on the given number of
// goroutines. Each index is handled by exactly one call and in the same
// order of operations as without parallelism, so the results do not depend
// on the number of workers.
func parallel(n, workers int, fn func(lo, hi int)) {
	var wg sync.WaitGroup
	chunk := (n + workers - 1) / workers
	for lo := 0; lo < n; lo += chunk {
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, min(lo+chunk, n))
	}
	wg.Wait()
}

// forwardRows computes the outputs lo to hi of layer l for b samples.
// Every weight row is used for all samples while it is in the cache.
func forwardRows(l *Layer, in, out []float64, b, lo, hi int) {
	for o := lo; o < hi; o++ {
		row, bias := l.Weights[o*l.Inputs:(o+1)*l.Inputs], l.Biases[o]
		for k := range b {
			x := in[k*l.Inputs : (k+1)*l.Inputs]
			if len(row) < 8 { // inline loop: a call costs more than tiny rows
				sum := bias
				for j, w := range row {
					sum += w * x[j]
				}
				out[k*l.Outputs+o] = sum
			} else {
				out[k*l.Outputs+o] = bias + dot(row, x)
			}
		}
	}
}

// gradientRows computes the weight and bias gradients of the outputs lo to
// hi of layer l, summed over b samples.
func gradientRows(l *Layer, in, delta, gw, gb []float64, b, lo, hi int) {
	for o := lo; o < hi; o++ {
		row := gw[o*l.Inputs : (o+1)*l.Inputs]
		clear(row)
		var sum float64
		for k := range b {
			d := delta[k*l.Outputs+o]
			sum += d
			x := in[k*l.Inputs : (k+1)*l.Inputs]
			if len(row) < 8 { // tiny rows: no call, no branch
				for j, a := range x {
					row[j] += d * a
				}
			} else if d != 0 { // skips the zeros of ReLU
				axpy(d, x, row)
			}
		}
		gb[o] = sum
	}
}

// backRows computes the deltas of the previous layer for the samples lo to
// hi: prev = Wᵀ·delta, multiplied by the derivative of its activation.
func backRows(l *Layer, prevAct Activation, in, delta, prev []float64, lo, hi int) {
	p := prev[lo*l.Inputs : hi*l.Inputs]
	clear(p)
	for o := range l.Outputs {
		row := l.Weights[o*l.Inputs : (o+1)*l.Inputs]
		for k := lo; k < hi; k++ {
			d := delta[k*l.Outputs+o]
			p := prev[k*l.Inputs : (k+1)*l.Inputs]
			if len(row) < 8 {
				for j, w := range row {
					p[j] += d * w
				}
			} else if d != 0 {
				axpy(d, row, p)
			}
		}
	}
	for j := range p {
		p[j] *= prevAct.derivative(in[lo*l.Inputs+j])
	}
}

// forwardBatch computes the activations of b samples stored in s.acts[0].
func (n *Network) forwardBatch(s *trainState, b, workers int) {
	for i, l := range n.Layers {
		in, out := s.acts[i][:b*l.Inputs], s.acts[i+1][:b*l.Outputs]
		if w := split(l.Outputs, workers, b*l.Inputs*l.Outputs); w > 1 {
			parallel(l.Outputs, w, func(lo, hi int) { forwardRows(l, in, out, b, lo, hi) })
		} else {
			forwardRows(l, in, out, b, 0, l.Outputs)
		}
		for k := range b {
			v := out[k*l.Outputs : (k+1)*l.Outputs]
			l.Activation.apply(v, v)
		}
	}
}

// Train performs one step of stochastic gradient descent on a single sample
// and returns the loss before the update.
func (n *Network) Train(x, target []float64, learningRate float64) (float64, error) {
	return n.TrainBatch([][]float64{x}, [][]float64{target}, learningRate)
}

// TrainBatch performs one step of gradient descent on a mini-batch and
// returns the mean loss before the update. The loss is cross-entropy for
// a Softmax output layer and half squared error otherwise.
func (n *Network) TrainBatch(xs, targets [][]float64, learningRate float64) (float64, error) {
	return n.TrainBatchWith(xs, targets, Step{Optimizer: SGD(), LearningRate: learningRate})
}

// Step configures a single update of TrainBatchWith.
type Step struct {
	Optimizer    Optimizer
	LearningRate float64
	// WeightDecay shrinks the weights (not the biases) by
	// LearningRate * WeightDecay * w on every step (decoupled L2 regularization).
	WeightDecay float64
	// Workers is the number of goroutines for large layers. Zero uses
	// GOMAXPROCS, one computes everything on the calling goroutine.
	Workers int
	// ClipNorm rescales the gradients so that their overall L2 norm is at
	// most ClipNorm, which keeps single outliers from blowing up the
	// weights. Zero disables clipping.
	ClipNorm float64
}

// TrainBatchWith performs one update on a mini-batch with the given
// optimizer and returns the mean loss before the update. If the loss or a
// gradient is not finite, the update is skipped and ErrNonFinite returned.
//
// Large layers are computed on several goroutines (see Step.Workers); the
// result is the same for any number of workers. A network must not be
// trained from several goroutines at the same time.
func (n *Network) TrainBatchWith(xs, targets [][]float64, step Step) (float64, error) {
	if len(xs) == 0 {
		return 0, errors.New("nn: empty batch")
	}
	if len(xs) != len(targets) {
		return 0, fmt.Errorf("nn: %d inputs but %d targets", len(xs), len(targets))
	}
	if step.Optimizer == nil {
		return 0, errors.New("nn: no optimizer")
	}
	in, out := n.InputSize(), n.OutputSize()
	for b, x := range xs {
		if len(x) != in {
			return 0, fmt.Errorf("%w: sample %d has %d values, network expects %d", ErrDimension, b, len(x), in)
		}
		if len(targets[b]) != out {
			return 0, fmt.Errorf("%w: target %d has %d values, network outputs %d", ErrDimension, b, len(targets[b]), out)
		}
	}
	b := len(xs)
	s := n.trainState(b)
	for k, x := range xs {
		copy(s.acts[0][k*in:], x)
	}
	var loss float64
	if n.small() {
		loss = n.trainSmall(s, b, targets)
	} else {
		n.forwardBatch(s, b, step.Workers)
		loss = n.outputDeltas(s, b, targets)
		n.backwardBatch(s, b, step.Workers)
	}

	// Mean gradients, optionally clipped. A non-finite loss or gradient
	// skips the update instead of destroying the weights.
	inv := 1 / float64(b)
	var norm float64
	for i := range n.Layers {
		for _, g := range s.gradW[i] {
			norm += g * g
		}
		for _, g := range s.gradB[i] {
			norm += g * g
		}
	}
	norm = math.Sqrt(norm) * inv
	if math.IsNaN(norm) || math.IsInf(norm, 0) || math.IsNaN(loss) || math.IsInf(loss, 0) {
		return loss * inv, ErrNonFinite
	}
	scale := inv
	if step.ClipNorm > 0 && norm > step.ClipNorm {
		scale *= step.ClipNorm / norm
	}
	for i, l := range n.Layers {
		for j := range s.gradW[i] {
			s.gradW[i][j] *= scale
		}
		for j := range s.gradB[i] {
			s.gradB[i][j] *= scale
		}
		if step.WeightDecay > 0 {
			shrink := 1 - step.LearningRate*step.WeightDecay
			for j := range l.Weights {
				l.Weights[j] *= shrink
			}
		}
		step.Optimizer.Step(2*i, l.Weights, s.gradW[i], step.LearningRate)
		step.Optimizer.Step(2*i+1, l.Biases, s.gradB[i], step.LearningRate)
	}
	return loss * inv, nil
}

// smallLayer is the size (inputs × outputs) up to which a layer counts as
// small. Networks of small layers train sample by sample: every vector fits
// into a few registers and the batch layout would only add overhead.
const smallLayer = 256

func (n *Network) small() bool {
	for _, l := range n.Layers {
		if l.Inputs*l.Outputs > smallLayer {
			return false
		}
	}
	return true
}

// trainSmall computes the gradients of a batch sample by sample and returns
// the summed loss. It gives the same result as the batched path.
func (n *Network) trainSmall(s *trainState, b int, targets [][]float64) float64 {
	for i := range n.Layers {
		clear(s.gradW[i])
		clear(s.gradB[i])
	}
	for k := range b {
		for i, l := range n.Layers {
			l.forward(s.acts[i][k*l.Inputs:(k+1)*l.Inputs], s.acts[i+1][k*l.Outputs:(k+1)*l.Outputs])
		}
	}
	loss := n.outputDeltas(s, b, targets)
	for k := range b {
		for i := len(n.Layers) - 1; i >= 0; i-- {
			l := n.Layers[i]
			in := s.acts[i][k*l.Inputs : (k+1)*l.Inputs]
			delta := s.deltas[i][k*l.Outputs : (k+1)*l.Outputs]
			gw, gb := s.gradW[i], s.gradB[i]
			for o, d := range delta {
				gb[o] += d
				row := gw[o*l.Inputs : (o+1)*l.Inputs]
				for j, a := range in {
					row[j] += d * a
				}
			}
			if i == 0 {
				break
			}
			prev := s.deltas[i-1][k*l.Inputs : (k+1)*l.Inputs]
			prevAct := n.Layers[i-1].Activation
			for j, a := range in {
				var sum float64
				for o, d := range delta {
					sum += l.Weights[o*l.Inputs+j] * d
				}
				prev[j] = sum * prevAct.derivative(a)
			}
		}
	}
	return loss
}

// outputDeltas computes the deltas of the output layer and returns the
// summed loss of the batch.
func (n *Network) outputDeltas(s *trainState, b int, targets [][]float64) float64 {
	last := len(n.Layers) - 1
	size := n.Layers[last].Outputs
	act := n.Layers[last].Activation
	outs, deltas := s.acts[last+1], s.deltas[last]
	var loss float64
	for k := range b {
		target := targets[k]
		for o, y := range outs[k*size : (k+1)*size] {
			diff := y - target[o]
			if act == Softmax {
				deltas[k*size+o] = diff
				if target[o] > 0 {
					loss -= target[o] * math.Log(max(y, 1e-15))
				}
			} else {
				deltas[k*size+o] = diff * act.derivative(y)
				loss += 0.5 * diff * diff
			}
		}
	}
	return loss
}

// backwardBatch computes the gradients of all layers from the output deltas.
func (n *Network) backwardBatch(s *trainState, b, workers int) {
	for i := len(n.Layers) - 1; i >= 0; i-- {
		l := n.Layers[i]
		in, delta := s.acts[i][:b*l.Inputs], s.deltas[i][:b*l.Outputs]
		gw, gb := s.gradW[i], s.gradB[i]
		work := b * l.Inputs * l.Outputs
		if w := split(l.Outputs, workers, work); w > 1 {
			parallel(l.Outputs, w, func(lo, hi int) { gradientRows(l, in, delta, gw, gb, b, lo, hi) })
		} else {
			gradientRows(l, in, delta, gw, gb, b, 0, l.Outputs)
		}
		if i == 0 {
			break
		}
		prev := s.deltas[i-1][:b*l.Inputs]
		prevAct := n.Layers[i-1].Activation
		if w := split(b, workers, work); w > 1 {
			parallel(b, w, func(lo, hi int) { backRows(l, prevAct, in, delta, prev, lo, hi) })
		} else {
			backRows(l, prevAct, in, delta, prev, 0, b)
		}
	}
}

// lossChunk is the number of samples Loss evaluates at once.
const lossChunk = 64

// Loss returns the mean loss of the network on the given samples without
// changing it: cross-entropy for a Softmax output layer, half squared error
// otherwise.
func (n *Network) Loss(xs, targets [][]float64) (float64, error) {
	if len(xs) != len(targets) {
		return 0, fmt.Errorf("nn: %d inputs but %d targets", len(xs), len(targets))
	}
	if len(xs) == 0 {
		return 0, nil
	}
	in, out := n.InputSize(), n.OutputSize()
	for i, x := range xs {
		if len(x) != in {
			return 0, fmt.Errorf("%w: input %d has %d values, network expects %d", ErrDimension, i, len(x), in)
		}
		if len(targets[i]) != out {
			return 0, fmt.Errorf("%w: target %d has %d values, network outputs %d", ErrDimension, i, len(targets[i]), out)
		}
	}
	s := n.trainState(min(len(xs), lossChunk))
	var loss float64
	for lo := 0; lo < len(xs); lo += lossChunk {
		hi := min(lo+lossChunk, len(xs))
		for k, x := range xs[lo:hi] {
			copy(s.acts[0][k*in:], x)
		}
		n.forwardBatch(s, hi-lo, 0)
		loss += n.outputDeltas(s, hi-lo, targets[lo:hi])
	}
	return loss / float64(len(xs)), nil
}
