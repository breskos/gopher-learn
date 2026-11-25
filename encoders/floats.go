package encoders

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/breskos/gopher-learn/internal/stats"
)

func requireFloats(in *Input) (int, error) {
	if in == nil || len(in.Values) == 0 {
		return 0, errors.New("no values")
	}
	if in.Type != Floats {
		return 0, errors.New("float model needs float input")
	}
	n := len(in.Values[0].Floats)
	for i, v := range in.Values {
		if len(v.Floats) != n {
			return 0, fmt.Errorf("value %d has %d dimensions, want %d", i, len(v.Floats), n)
		}
	}
	return n, nil
}

// FloatExactModel passes float slices through unchanged.
type FloatExactModel struct {
	Size int `json:"size"`
}

// Fit implements Model.
func (m *FloatExactModel) Fit(in *Input, _ Config) error {
	n, err := requireFloats(in)
	m.Size = n
	return err
}

// Encode implements Model.
func (m *FloatExactModel) Encode(v Value) ([]float64, error) {
	if len(v.Floats) != m.Size {
		return nil, fmt.Errorf("encoders: got %d floats, want %d", len(v.Floats), m.Size)
	}
	return slices.Clone(v.Floats), nil
}

// Dimensions implements Model.
func (m *FloatExactModel) Dimensions() int { return m.Size }

// FloatReducerModel removes float dimensions that carry little information:
// dimensions with a constant value, and dimensions whose absolute Spearman
// rank correlation with an earlier kept dimension reaches
// Config.FloatReducerCorrelation.
type FloatReducerModel struct {
	// Keep marks the input dimensions that are kept, in input order.
	Keep []bool `json:"keep"`
}

// Fit implements Model.
func (m *FloatReducerModel) Fit(in *Input, cfg Config) error {
	n, err := requireFloats(in)
	if err != nil {
		return err
	}
	cols := make([][]float64, n)
	for _, v := range in.Values {
		for i, x := range v.Floats {
			cols[i] = append(cols[i], x)
		}
	}
	m.Keep = make([]bool, n)
	var kept []int
	for i, col := range cols {
		if isConstant(col) {
			continue
		}
		redundant := false
		for _, j := range kept {
			if rs := stats.Spearman(cols[j], col); !math.IsNaN(rs) && math.Abs(rs) >= cfg.FloatReducerCorrelation {
				redundant = true
				break
			}
		}
		if !redundant {
			m.Keep[i] = true
			kept = append(kept, i)
		}
	}
	return nil
}

// Encode implements Model.
func (m *FloatReducerModel) Encode(v Value) ([]float64, error) {
	if len(v.Floats) != len(m.Keep) {
		return nil, fmt.Errorf("encoders: got %d floats, want %d", len(v.Floats), len(m.Keep))
	}
	out := make([]float64, 0, m.Dimensions())
	for i, keep := range m.Keep {
		if keep {
			out = append(out, v.Floats[i])
		}
	}
	return out, nil
}

// Dimensions implements Model.
func (m *FloatReducerModel) Dimensions() int {
	n := 0
	for _, k := range m.Keep {
		if k {
			n++
		}
	}
	return n
}

func isConstant(v []float64) bool {
	for _, x := range v[1:] {
		if x != v[0] {
			return false
		}
	}
	return true
}
