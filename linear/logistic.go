package linear

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/internal/optim"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

// Logistic is a multinomial logistic regression (softmax regression): the
// probability of class k is proportional to exp(Weights[k] · x + Biases[k]).
// With two classes it is the classic logistic regression.
type Logistic struct {
	Labels []string `json:"labels"`
	// Weights holds one row per class in the units of the original
	// features: a larger weight raises the probability of that class.
	Weights [][]float64 `json:"weights"`
	Biases  []float64   `json:"biases"`
	// Features names the inputs, if the training set had names.
	Features []string `json:"features,omitempty"`
	// Iterations and Converged describe the optimization.
	Iterations int  `json:"iterations"`
	Converged  bool `json:"converged"`
}

// LogisticConfig configures FitLogistic.
type LogisticConfig struct {
	// L2 is the penalty on the standardized weights. Zero uses 1e-4, which
	// keeps the weights finite on separable data; a negative value turns
	// the penalty off. Larger values (0.01 to 1) regularize more.
	L2 float64
	// MaxIter bounds the number of L-BFGS iterations. Default 500.
	MaxIter int
	// Tolerance stops the optimization when the largest gradient entry
	// falls below it. Default 1e-6.
	Tolerance float64
}

// FitLogistic fits a logistic regression by minimizing the mean
// cross-entropy plus the L2 penalty with L-BFGS. The loss is convex, so the
// result does not depend on a random start.
func FitLogistic(set *learn.Set, cfg LogisticConfig) (*Logistic, error) {
	if err := checkSet(set, nn.Classification); err != nil {
		return nil, err
	}
	if len(set.Labels) < 2 {
		return nil, errors.New("linear: logistic regression needs at least two classes")
	}
	l2 := cfg.L2
	switch {
	case math.IsNaN(l2):
		return nil, errors.New("linear: L2 is NaN")
	case l2 == 0:
		l2 = 1e-4
	case l2 < 0:
		l2 = 0
	}
	mean, inv, err := standardization(set)
	if err != nil {
		return nil, err
	}
	n, d, k := set.Len(), set.InputSize(), len(set.Labels)
	z := make([]float64, n*d)
	for i, s := range set.Samples {
		for j, v := range s.Vector {
			z[i*d+j] = (v - mean[j]) * inv[j]
		}
	}
	stride := d + 1 // per class: d weights and a bias
	partLoss := make([]float64, maxChunks)
	partGrad := make([][]float64, maxChunks)
	objective := func(theta, grad []float64) float64 {
		chunks := parallelRows(n, n*d*k, func(c, lo, hi int) {
			g := partGrad[c]
			if g == nil {
				g = make([]float64, len(theta))
				partGrad[c] = g
			}
			clear(g)
			p := make([]float64, k)
			var loss float64
			for i := lo; i < hi; i++ {
				x := z[i*d : (i+1)*d]
				m := math.Inf(-1)
				for c := range k {
					w := theta[c*stride : c*stride+d]
					v := theta[c*stride+d]
					for j, xv := range x {
						v += w[j] * xv
					}
					p[c] = v
					m = max(m, v)
				}
				var sum float64
				for c := range k {
					p[c] = math.Exp(p[c] - m)
					sum += p[c]
				}
				y := set.Samples[i].Class
				loss -= math.Log(max(p[y]/sum, 1e-300))
				for c := range k {
					r := p[c] / sum
					if c == y {
						r--
					}
					gw := g[c*stride : c*stride+d]
					for j, xv := range x {
						gw[j] += r * xv
					}
					g[c*stride+d] += r
				}
			}
			partLoss[c] = loss
		})
		clear(grad)
		var loss float64
		for c := range chunks {
			loss += partLoss[c]
			for i, v := range partGrad[c] {
				grad[i] += v
			}
		}
		inv := 1 / float64(n)
		loss *= inv
		for i := range grad {
			grad[i] *= inv
		}
		for c := range k {
			w := theta[c*stride : c*stride+d]
			for j, v := range w {
				loss += 0.5 * l2 * v * v
				grad[c*stride+j] += l2 * v
			}
		}
		return loss
	}
	res, err := optim.LBFGS(objective, make([]float64, k*stride), optim.Config{MaxIter: cfg.MaxIter, GradTol: cfg.Tolerance})
	if err != nil {
		return nil, fmt.Errorf("linear: %w", err)
	}
	m := &Logistic{
		Labels: slices.Clone(set.Labels), Weights: make([][]float64, k), Biases: make([]float64, k),
		Features: featureNames(set), Iterations: res.Iterations, Converged: res.Converged,
	}
	for c := range k {
		m.Weights[c] = make([]float64, d)
		m.Biases[c] = res.X[c*stride+d]
		for j := range d {
			m.Weights[c][j] = res.X[c*stride+j] * inv[j]
			m.Biases[c] -= m.Weights[c][j] * mean[j]
		}
	}
	return m, nil
}

// Validate checks the model, e.g. after decoding it.
func (m *Logistic) Validate() error {
	k := len(m.Labels)
	if k < 2 || len(m.Weights) != k || len(m.Biases) != k {
		return fmt.Errorf("linear: logistic regression with %d labels, %d weight rows and %d biases", k, len(m.Weights), len(m.Biases))
	}
	d := len(m.Weights[0])
	if d == 0 || len(m.Features) > 0 && len(m.Features) != d {
		return errors.New("linear: logistic regression has no weights or wrong feature names")
	}
	for c, w := range m.Weights {
		if len(w) != d {
			return fmt.Errorf("linear: weight row %d has %d values, want %d", c, len(w), d)
		}
		for _, v := range append(slices.Clone(w), m.Biases[c]) {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return errors.New("linear: non-finite coefficient")
			}
		}
	}
	return nil
}

// ProbabilitiesInto writes the class probabilities of x into dst, which
// must have one entry per label, without allocating.
func (m *Logistic) ProbabilitiesInto(dst, x []float64) error {
	if len(dst) != len(m.Labels) {
		return fmt.Errorf("%w: dst has %d values for %d classes", nn.ErrDimension, len(dst), len(m.Labels))
	}
	if len(x) != len(m.Weights[0]) {
		return fmt.Errorf("%w: input has %d values, model expects %d", nn.ErrDimension, len(x), len(m.Weights[0]))
	}
	mx := math.Inf(-1)
	for c, w := range m.Weights {
		v := m.Biases[c]
		for j, wj := range w {
			v += wj * x[j]
		}
		dst[c] = v
		mx = max(mx, v)
	}
	var sum float64
	for c := range dst {
		dst[c] = math.Exp(dst[c] - mx)
		sum += dst[c]
	}
	for c := range dst {
		dst[c] /= sum
	}
	return nil
}

// Probabilities returns the class probabilities of x in the order of Labels.
func (m *Logistic) Probabilities(x []float64) ([]float64, error) {
	p := make([]float64, len(m.Labels))
	if err := m.ProbabilitiesInto(p, x); err != nil {
		return nil, err
	}
	return p, nil
}

// Classify returns the most probable class of x.
func (m *Logistic) Classify(x []float64) (string, error) {
	p, err := m.Probabilities(x)
	if err != nil {
		return "", err
	}
	return m.Labels[nn.ArgMax(p)], nil
}

// Evaluate classifies every sample of a classification set whose labels
// match the model's.
func (m *Logistic) Evaluate(set *learn.Set) (*evaluation.Evaluation, error) {
	if !slices.Equal(set.Labels, m.Labels) {
		return nil, fmt.Errorf("linear: set labels %v differ from model labels %v", set.Labels, m.Labels)
	}
	e := evaluation.New(nn.Classification, m.Labels)
	p := make([]float64, len(m.Labels))
	for i, s := range set.Samples {
		if err := m.ProbabilitiesInto(p, s.Vector); err != nil {
			return nil, fmt.Errorf("linear: sample %d: %w", i, err)
		}
		e.Add(s.Label, m.Labels[nn.ArgMax(p)])
		if err := e.AddScores(p, s.Class); err != nil {
			return nil, err
		}
	}
	return e, nil
}
