package encoders

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"slices"
)

// HashingModel encodes the tokens of a string with the hashing trick: every
// token adds +1 or -1 (decided by the hash, so collisions tend to cancel out)
// to one of Size dimensions. It has no vocabulary, so it keeps working on
// words it has never seen, and its size is fixed, which suits streams.
type HashingModel struct {
	Size int `json:"size"`
}

// Fit implements Model. It only sets the size from Config.HashDimensions.
func (m *HashingModel) Fit(in *Input, cfg Config) error {
	if err := requireStrings(in); err != nil {
		return err
	}
	if cfg.HashDimensions < 1 {
		return errors.New("HashDimensions must be positive")
	}
	m.Size = cfg.HashDimensions
	return nil
}

// Encode implements Model.
func (m *HashingModel) Encode(v Value) ([]float64, error) {
	out := make([]float64, m.Size)
	if m.Size == 0 {
		return out, nil
	}
	for _, t := range tokenize(v.String) {
		h := fnv.New64a()
		_, _ = h.Write([]byte(t))
		sum := h.Sum64()
		sign := 1.0
		if sum>>63 == 1 {
			sign = -1
		}
		out[sum%uint64(m.Size)] += sign
	}
	return out, nil
}

// Dimensions implements Model.
func (m *HashingModel) Dimensions() int { return m.Size }

// TFIDFModel encodes a string as the term frequencies of its tokens weighted
// by their inverse document frequency in the fitted texts, normalized to
// unit length. Frequent, uninformative words get low weights.
type TFIDFModel struct {
	Terms []string  `json:"terms"`
	IDF   []float64 `json:"idf"`

	lookup map[string]int
}

// Fit implements Model.
func (m *TFIDFModel) Fit(in *Input, cfg Config) error {
	if err := requireStrings(in); err != nil {
		return err
	}
	if cfg.TFIDFMaxTerms < 1 {
		return errors.New("TFIDFMaxTerms must be positive")
	}
	df := map[string]int{}
	for _, v := range in.Values {
		seen := map[string]bool{}
		for _, t := range tokenize(v.String) {
			if !seen[t] {
				seen[t] = true
				df[t]++
			}
		}
	}
	terms := make([]string, 0, len(df))
	for t := range df {
		terms = append(terms, t)
	}
	// most frequent first, ties alphabetically for a stable vocabulary
	slices.SortFunc(terms, func(a, b string) int {
		if c := cmp.Compare(df[b], df[a]); c != 0 {
			return c
		}
		return cmp.Compare(a, b)
	})
	terms = terms[:min(len(terms), cfg.TFIDFMaxTerms)]
	slices.Sort(terms)
	n := float64(len(in.Values))
	m.Terms, m.IDF = terms, make([]float64, len(terms))
	for i, t := range terms {
		m.IDF[i] = math.Log((1+n)/(1+float64(df[t]))) + 1
	}
	m.lookup = indexOf(m.Terms)
	return nil
}

// Encode implements Model.
func (m *TFIDFModel) Encode(v Value) ([]float64, error) {
	out := make([]float64, len(m.Terms))
	for _, t := range tokenize(v.String) {
		if i, ok := m.lookup[t]; ok {
			out[i] += m.IDF[i]
		}
	}
	var norm float64
	for _, x := range out {
		norm += x * x
	}
	if norm > 0 {
		norm = math.Sqrt(norm)
		for i := range out {
			out[i] /= norm
		}
	}
	return out, nil
}

// Dimensions implements Model.
func (m *TFIDFModel) Dimensions() int { return len(m.Terms) }

// UnmarshalJSON decodes the model and rebuilds its lookup table.
func (m *TFIDFModel) UnmarshalJSON(b []byte) error {
	type plain TFIDFModel
	if err := json.Unmarshal(b, (*plain)(m)); err != nil {
		return err
	}
	if len(m.IDF) != len(m.Terms) {
		return fmt.Errorf("encoders: %d terms but %d weights", len(m.Terms), len(m.IDF))
	}
	m.lookup = indexOf(m.Terms)
	return nil
}

// FloatCategoriesModel one-hot encodes every dimension of a float vector
// whose values are category codes. Values not seen during fitting encode
// to zeros.
type FloatCategoriesModel struct {
	// Values holds the sorted distinct values per input dimension.
	Values [][]float64 `json:"values"`
}

// Fit implements Model.
func (m *FloatCategoriesModel) Fit(in *Input, cfg Config) error {
	n, err := requireFloats(in)
	if err != nil {
		return err
	}
	m.Values = make([][]float64, n)
	for d := range n {
		seen := map[float64]bool{}
		for _, v := range in.Values {
			x := v.Floats[d]
			if math.IsNaN(x) || math.IsInf(x, 0) {
				return fmt.Errorf("dimension %d: non-finite value %v", d, x)
			}
			if !seen[x] {
				seen[x] = true
				m.Values[d] = append(m.Values[d], x)
			}
		}
		if len(m.Values[d]) > cfg.CategoriesMax {
			return fmt.Errorf("dimension %d has %d distinct values, more than CategoriesMax (%d)", d, len(m.Values[d]), cfg.CategoriesMax)
		}
		slices.Sort(m.Values[d])
	}
	return nil
}

// Encode implements Model.
func (m *FloatCategoriesModel) Encode(v Value) ([]float64, error) {
	if len(v.Floats) != len(m.Values) {
		return nil, fmt.Errorf("encoders: got %d floats, want %d", len(v.Floats), len(m.Values))
	}
	out := make([]float64, 0, m.Dimensions())
	for d, x := range v.Floats {
		block := make([]float64, len(m.Values[d]))
		if i, ok := slices.BinarySearch(m.Values[d], x); ok {
			block[i] = 1
		}
		out = append(out, block...)
	}
	return out, nil
}

// Dimensions implements Model.
func (m *FloatCategoriesModel) Dimensions() int {
	n := 0
	for _, v := range m.Values {
		n += len(v)
	}
	return n
}
