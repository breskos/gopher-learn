package nn

import (
	"errors"
	"fmt"
	"math"
)

// trainState holds reusable buffers for backpropagation.
type trainState struct {
	acts   [][]float64 // acts[0] is the input, acts[i+1] the output of layer i
	deltas [][]float64
	gradW  [][]float64
	gradB  [][]float64
}

func (n *Network) trainState() *trainState {
	if s := n.state; s != nil && len(s.gradW) == len(n.Layers) {
		return s
	}
	s := &trainState{
		acts:   make([][]float64, len(n.Layers)+1),
		deltas: make([][]float64, len(n.Layers)),
		gradW:  make([][]float64, len(n.Layers)),
		gradB:  make([][]float64, len(n.Layers)),
	}
	for i, l := range n.Layers {
		s.acts[i+1] = make([]float64, l.Outputs)
		s.deltas[i] = make([]float64, l.Outputs)
		s.gradW[i] = make([]float64, len(l.Weights))
		s.gradB[i] = make([]float64, l.Outputs)
	}
	n.state = s
	return s
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
}

// TrainBatchWith performs one update on a mini-batch with the given
// optimizer and returns the mean loss before the update.
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
	s := n.trainState()
	for i := range n.Layers {
		clear(s.gradW[i])
		clear(s.gradB[i])
	}

	var loss float64
	for b, x := range xs {
		if len(x) != n.InputSize() {
			return 0, fmt.Errorf("%w: sample %d has %d values, network expects %d", ErrDimension, b, len(x), n.InputSize())
		}
		if len(targets[b]) != n.OutputSize() {
			return 0, fmt.Errorf("%w: target %d has %d values, network outputs %d", ErrDimension, b, len(targets[b]), n.OutputSize())
		}
		loss += n.backprop(s, x, targets[b])
	}

	inv := 1 / float64(len(xs))
	for i, l := range n.Layers {
		for j := range s.gradW[i] {
			s.gradW[i][j] *= inv
		}
		for j := range s.gradB[i] {
			s.gradB[i][j] *= inv
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
	softmax := n.Layers[len(n.Layers)-1].Activation == Softmax
	var loss float64
	for i, x := range xs {
		out, err := n.Predict(x)
		if err != nil {
			return 0, err
		}
		if len(targets[i]) != len(out) {
			return 0, fmt.Errorf("%w: target %d has %d values, network outputs %d", ErrDimension, i, len(targets[i]), len(out))
		}
		for o, y := range out {
			t := targets[i][o]
			if softmax {
				if t > 0 {
					loss -= t * math.Log(max(y, 1e-15))
				}
			} else {
				loss += 0.5 * (y - t) * (y - t)
			}
		}
	}
	return loss / float64(len(xs)), nil
}

// backprop accumulates the gradients of one sample into s and returns its loss.
func (n *Network) backprop(s *trainState, x, target []float64) float64 {
	s.acts[0] = x
	for i, l := range n.Layers {
		l.forward(s.acts[i], s.acts[i+1])
	}

	last := len(n.Layers) - 1
	out := s.acts[last+1]
	outAct := n.Layers[last].Activation
	var loss float64
	for o, y := range out {
		diff := y - target[o]
		if outAct == Softmax {
			s.deltas[last][o] = diff
			if target[o] > 0 {
				loss -= target[o] * math.Log(max(y, 1e-15))
			}
		} else {
			s.deltas[last][o] = diff * outAct.derivative(y)
			loss += 0.5 * diff * diff
		}
	}

	for i := last; i >= 0; i-- {
		l := n.Layers[i]
		in := s.acts[i]
		delta := s.deltas[i]
		gw := s.gradW[i]
		for o, d := range delta {
			s.gradB[i][o] += d
			row := gw[o*l.Inputs : (o+1)*l.Inputs]
			for j, a := range in {
				row[j] += d * a
			}
		}
		if i == 0 {
			break
		}
		prev := s.deltas[i-1]
		prevAct := n.Layers[i-1].Activation
		for j := range prev {
			var sum float64
			for o, d := range delta {
				sum += l.Weights[o*l.Inputs+j] * d
			}
			prev[j] = sum * prevAct.derivative(in[j])
		}
	}
	return loss
}
