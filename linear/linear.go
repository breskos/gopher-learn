// Package linear provides linear and logistic regression: fast, exact,
// interpretable baselines for tabular data.
//
// Both models standardize the features internally, so the regularization
// treats all features alike and training is numerically stable, and they
// report their coefficients in the units of the original features.
//
//	reg, err := linear.FitRegression(set, linear.RegressionConfig{L2: 0.01})
//	y, err := reg.Predict(x)
//
//	clf, err := linear.FitLogistic(set, linear.LogisticConfig{})
//	label, err := clf.Classify(x)
package linear

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"slices"
	"sync"

	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

// standardization returns the mean and the inverse standard deviation of
// every feature (1 for constant features).
func standardization(set *learn.Set) (mean, inv []float64, err error) {
	vectors := make([][]float64, set.Len())
	for i, s := range set.Samples {
		vectors[i] = s.Vector
	}
	sc, err := learn.FitScalerVectors(vectors, learn.Standardize)
	if err != nil {
		return nil, nil, err
	}
	return sc.Offset, sc.Factor, nil
}

func checkSet(set *learn.Set, task nn.Task) error {
	if set == nil || set.Len() < 2 {
		return errors.New("linear: need at least two samples")
	}
	if set.Task != task {
		return fmt.Errorf("linear: need a %v set, got %v", task, set.Task)
	}
	return set.Validate(0, 0)
}

func featureNames(set *learn.Set) []string {
	if len(set.Features) == set.InputSize() {
		return slices.Clone(set.Features)
	}
	return nil
}

// maxChunks is the number of chunks large computations are split into.
const maxChunks = 64

// parallelRows splits [0, n) into chunks and calls fn on them, on several
// goroutines when the work is large enough. The chunks depend only on n and
// the work, not on the machine, so callers that combine per-chunk results
// in chunk order get bit-identical results everywhere. It returns the
// number of chunks.
func parallelRows(n, work int, fn func(chunk, lo, hi int)) int {
	chunks := 1
	if work >= 1<<16 {
		chunks = min(maxChunks, n)
	}
	size := (n + chunks - 1) / chunks
	chunks = (n + size - 1) / size
	if chunks == 1 {
		fn(0, 0, n)
		return 1
	}
	var wg sync.WaitGroup
	next := make(chan int, chunks)
	for c := range chunks {
		next <- c
	}
	close(next)
	for range min(runtime.GOMAXPROCS(0), chunks) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for c := range next {
				fn(c, c*size, min((c+1)*size, n))
			}
		}()
	}
	wg.Wait()
	return chunks
}

// Regression is a linear regression model: y = Weights · x + Bias.
type Regression struct {
	// Weights are the coefficients in the units of the original features:
	// the change of y per unit of the feature.
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
	// Features names the inputs, if the training set had names.
	Features []string `json:"features,omitempty"`
}

// RegressionConfig configures FitRegression.
type RegressionConfig struct {
	// L2 is the ridge penalty on the standardized weights. Zero fits plain
	// least squares; values like 0.01 to 1 shrink the weights and help
	// with correlated features.
	L2 float64
}

// FitRegression fits a linear regression with an optional ridge penalty by
// solving the normal equations exactly (Cholesky decomposition).
func FitRegression(set *learn.Set, cfg RegressionConfig) (*Regression, error) {
	if err := checkSet(set, nn.Regression); err != nil {
		return nil, err
	}
	if cfg.L2 < 0 || math.IsNaN(cfg.L2) {
		return nil, fmt.Errorf("linear: L2 must not be negative, got %v", cfg.L2)
	}
	mean, inv, err := standardization(set)
	if err != nil {
		return nil, err
	}
	n, d := set.Len(), set.InputSize()
	var ymean float64
	for _, s := range set.Samples {
		ymean += s.Value
	}
	ymean /= float64(n)

	// A = Zᵀ Z / n + L2·I and b = Zᵀ (y - ȳ) / n, accumulated per chunk of
	// samples and summed in chunk order
	partA := make([][]float64, maxChunks)
	partB := make([][]float64, maxChunks)
	chunks := parallelRows(n, n*d*d, func(c, lo, hi int) {
		a, b := make([]float64, d*d), make([]float64, d)
		z := make([]float64, d)
		for _, s := range set.Samples[lo:hi] {
			for j, v := range s.Vector {
				z[j] = (v - mean[j]) * inv[j]
			}
			y := s.Value - ymean
			for i, zi := range z {
				b[i] += zi * y
				row := a[i*d : i*d+i+1]
				for j := range row {
					row[j] += zi * z[j]
				}
			}
		}
		partA[c], partB[c] = a, b
	})
	A, b := make([]float64, d*d), make([]float64, d)
	for c := range chunks {
		for i, v := range partA[c] {
			A[i] += v
		}
		for i, v := range partB[c] {
			b[i] += v
		}
	}
	for i := range d {
		for j := 0; j <= i; j++ {
			A[i*d+j] /= float64(n)
			A[j*d+i] = A[i*d+j]
		}
		A[i*d+i] += cfg.L2
		b[i] /= float64(n)
	}
	w, err := solveSPD(A, b, d)
	if err != nil {
		return nil, err
	}
	// back to the units of the original features
	r := &Regression{Weights: make([]float64, d), Bias: ymean, Features: featureNames(set)}
	for j := range w {
		r.Weights[j] = w[j] * inv[j]
		r.Bias -= r.Weights[j] * mean[j]
	}
	return r, nil
}

// solveSPD solves A·x = b for a symmetric positive (semi-)definite A with
// a Cholesky decomposition. A singular A (e.g. duplicated features without
// a penalty) gets a tiny ridge added so that a solution still exists.
func solveSPD(A, b []float64, d int) ([]float64, error) {
	trace := 0.0
	for i := range d {
		trace += A[i*d+i]
	}
	jitter := 0.0
	for attempt := 0; attempt < 8; attempt++ {
		L := slices.Clone(A)
		for i := range d {
			L[i*d+i] += jitter
		}
		if cholesky(L, d) {
			// forward and back substitution
			x := slices.Clone(b)
			for i := range d {
				for k := range i {
					x[i] -= L[i*d+k] * x[k]
				}
				x[i] /= L[i*d+i]
			}
			for i := d - 1; i >= 0; i-- {
				for k := i + 1; k < d; k++ {
					x[i] -= L[k*d+i] * x[k]
				}
				x[i] /= L[i*d+i]
			}
			return x, nil
		}
		jitter = max(jitter*100, 1e-10*max(trace/float64(d), 1))
	}
	return nil, errors.New("linear: the normal equations cannot be solved")
}

// cholesky overwrites the lower triangle of A with L, where A = L·Lᵀ, and
// reports whether A is positive definite.
func cholesky(A []float64, d int) bool {
	for j := range d {
		s := A[j*d+j]
		for k := range j {
			s -= A[j*d+k] * A[j*d+k]
		}
		if s <= 0 || math.IsNaN(s) {
			return false
		}
		A[j*d+j] = math.Sqrt(s)
		for i := j + 1; i < d; i++ {
			v := A[i*d+j]
			for k := range j {
				v -= A[i*d+k] * A[j*d+k]
			}
			A[i*d+j] = v / A[j*d+j]
		}
	}
	return true
}

// Validate checks the model, e.g. after decoding it.
func (r *Regression) Validate() error {
	if len(r.Weights) == 0 {
		return errors.New("linear: regression without weights")
	}
	if len(r.Features) > 0 && len(r.Features) != len(r.Weights) {
		return fmt.Errorf("linear: %d feature names for %d weights", len(r.Features), len(r.Weights))
	}
	for _, w := range append(slices.Clone(r.Weights), r.Bias) {
		if math.IsNaN(w) || math.IsInf(w, 0) {
			return errors.New("linear: non-finite coefficient")
		}
	}
	return nil
}

// Predict returns the prediction for x.
func (r *Regression) Predict(x []float64) (float64, error) {
	if len(x) != len(r.Weights) {
		return 0, fmt.Errorf("%w: input has %d values, model expects %d", nn.ErrDimension, len(x), len(r.Weights))
	}
	y := r.Bias
	for j, w := range r.Weights {
		y += w * x[j]
	}
	return y, nil
}

// Evaluate predicts every sample of a regression set. threshold is the
// largest error that counts as correct.
func (r *Regression) Evaluate(set *learn.Set, threshold float64) (*evaluation.Evaluation, error) {
	e := evaluation.New(nn.Regression, nil)
	e.Threshold = threshold
	for i, s := range set.Samples {
		y, err := r.Predict(s.Vector)
		if err != nil {
			return nil, fmt.Errorf("linear: sample %d: %w", i, err)
		}
		e.AddRegression(s.Value, y)
	}
	return e, nil
}
