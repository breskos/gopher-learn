package learn

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/breskos/gopher-learn/nn"
)

// ScaleMethod selects how a Scaler is fitted.
type ScaleMethod int

const (
	// Standardize scales every feature to zero mean and unit variance.
	Standardize ScaleMethod = iota
	// MinMax scales every feature to the range [0, 1].
	MinMax
)

var scaleMethodNames = [...]string{"standardize", "min-max"}

// String returns the name of the method.
func (m ScaleMethod) String() string {
	if m < 0 || int(m) >= len(scaleMethodNames) {
		return fmt.Sprintf("ScaleMethod(%d)", int(m))
	}
	return scaleMethodNames[m]
}

// MarshalText implements encoding.TextMarshaler.
func (m ScaleMethod) MarshalText() ([]byte, error) {
	if m < 0 || int(m) >= len(scaleMethodNames) {
		return nil, fmt.Errorf("learn: invalid scale method %d", int(m))
	}
	return []byte(scaleMethodNames[m]), nil
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (m *ScaleMethod) UnmarshalText(b []byte) error {
	i := slices.Index(scaleMethodNames[:], string(b))
	if i < 0 {
		return fmt.Errorf("learn: unknown scale method %q", b)
	}
	*m = ScaleMethod(i)
	return nil
}

// Scaler rescales feature vectors: x' = (x - Offset) * Factor, per feature.
//
// Fit a scaler on the training data only and apply the same scaler to
// validation, test and production data. Scalers serialize to JSON.
type Scaler struct {
	Method ScaleMethod `json:"method"`
	Offset []float64   `json:"offset"`
	Factor []float64   `json:"factor"`
}

// FitScaler computes a scaler from the feature vectors of a set. Features
// with a constant value are only shifted, never divided by zero.
func FitScaler(s *Set, method ScaleMethod) (*Scaler, error) {
	if s == nil || s.Len() == 0 {
		return nil, errors.New("learn: cannot fit a scaler on an empty set")
	}
	vectors := make([][]float64, len(s.Samples))
	for i, sm := range s.Samples {
		vectors[i] = sm.Vector
	}
	return FitScalerVectors(vectors, method)
}

// FitScalerVectors computes a scaler from feature vectors of equal length.
func FitScalerVectors(vectors [][]float64, method ScaleMethod) (*Scaler, error) {
	if len(vectors) == 0 || len(vectors[0]) == 0 {
		return nil, errors.New("learn: cannot fit a scaler on no data")
	}
	n := len(vectors[0])
	for i, v := range vectors {
		if len(v) != n {
			return nil, fmt.Errorf("%w: vector %d has %d features, want %d", nn.ErrDimension, i, len(v), n)
		}
	}
	sc := &Scaler{Method: method, Offset: make([]float64, n), Factor: make([]float64, n)}
	switch method {
	case Standardize:
		// Welford's algorithm for a numerically stable mean and variance.
		m2 := make([]float64, n)
		for k, v := range vectors {
			for i, x := range v {
				d := x - sc.Offset[i]
				sc.Offset[i] += d / float64(k+1)
				m2[i] += d * (x - sc.Offset[i])
			}
		}
		for i := range n {
			std := math.Sqrt(m2[i] / float64(len(vectors)))
			sc.Factor[i] = inverse(std)
		}
	case MinMax:
		lo := slices.Clone(vectors[0])
		hi := slices.Clone(lo)
		for _, v := range vectors[1:] {
			for i, x := range v {
				lo[i], hi[i] = min(lo[i], x), max(hi[i], x)
			}
		}
		for i := range n {
			sc.Offset[i] = lo[i]
			sc.Factor[i] = inverse(hi[i] - lo[i])
		}
	default:
		return nil, fmt.Errorf("learn: invalid scale method %v", method)
	}
	return sc, nil
}

func inverse(spread float64) float64 {
	if spread == 0 || math.IsNaN(spread) {
		return 1
	}
	return 1 / spread
}

// Validate checks that the scaler is consistent, e.g. after decoding.
func (sc *Scaler) Validate() error {
	if len(sc.Offset) == 0 || len(sc.Offset) != len(sc.Factor) {
		return fmt.Errorf("learn: scaler has %d offsets and %d factors", len(sc.Offset), len(sc.Factor))
	}
	for i := range sc.Offset {
		if !finite(sc.Offset[i]) || !finite(sc.Factor[i]) || sc.Factor[i] == 0 {
			return fmt.Errorf("learn: scaler has invalid values for feature %d", i)
		}
	}
	return nil
}

// Size returns the number of features the scaler expects.
func (sc *Scaler) Size() int { return len(sc.Offset) }

// Transform returns a scaled copy of v.
func (sc *Scaler) Transform(v []float64) ([]float64, error) {
	if len(v) != sc.Size() {
		return nil, fmt.Errorf("%w: vector has %d features, scaler expects %d", nn.ErrDimension, len(v), sc.Size())
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = (x - sc.Offset[i]) * sc.Factor[i]
	}
	return out, nil
}

// Inverse undoes Transform.
func (sc *Scaler) Inverse(v []float64) ([]float64, error) {
	if len(v) != sc.Size() {
		return nil, fmt.Errorf("%w: vector has %d features, scaler expects %d", nn.ErrDimension, len(v), sc.Size())
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x/sc.Factor[i] + sc.Offset[i]
	}
	return out, nil
}

// TransformSet returns a new set with scaled copies of all samples. The
// original set is not modified.
func (sc *Scaler) TransformSet(s *Set) (*Set, error) {
	out := &Set{Task: s.Task, Labels: slices.Clone(s.Labels), Features: slices.Clone(s.Features), Samples: make([]*Sample, len(s.Samples))}
	for i, sm := range s.Samples {
		v, err := sc.Transform(sm.Vector)
		if err != nil {
			return nil, fmt.Errorf("learn: sample %d: %w", i, err)
		}
		c := *sm
		c.Vector = v
		c.Output = slices.Clone(sm.Output)
		out.Samples[i] = &c
	}
	return out, nil
}

func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }
