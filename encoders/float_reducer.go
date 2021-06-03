package encoders

import (
	"log"
	"math"

	"github.com/breskos/gopher-learn/analysis"
)

/*
FloatReducer contains of threee parts.
It first runs Spearman correlation. If the correlcation of two dimensions is equal or
higher than the defined threshold (in config) on of the dimensions is cut away.
This can be beneficial but also cause problems (in the case that a variable is despite
the high correlation highly important).
Also the FloatrReducer cuts away dimensions that just have one value (and therefore add no information gain).
*/

type FloatReducerModel struct {
	Model      map[int]bool
	Dimensions int
	Quality    float64
	Config     *EncoderConfig
}

func NewFloatReducerModel(config *EncoderConfig) *FloatReducerModel {
	return &FloatReducerModel{
		Model:  make(map[int]bool),
		Config: config,
	}
}

func (m *FloatReducerModel) Fit(set *Input) {
	if len(set.Values) < 1 {
		log.Fatalf("no values delivered for fit")
	}
	m.Model = make(map[int]bool)
	spearman := make(map[int]map[int]float64)
	dimensions := make([][]float64, len(set.Values[0].Float))
	for _, sample := range set.Values {
		for i, x := range sample.Float {
			dimensions[i] = append(dimensions[i], x)
		}
	}
	for i := range dimensions {
		spearman[i] = make(map[int]float64)
		for j := range dimensions {
			if i != j {
				rs, _ := analysis.Spearman(dimensions[i], dimensions[j])
				spearman[i][j] = rs
				if math.Abs(rs) > math.Abs(m.Config.FloatReducerSpearman) {
					m.Model[i] = false
				}
			}
		}
		m.Model[i] = true
		if similarValues(dimensions[i]) {
			m.Model[i] = false
		}
	}
	for i := range spearman {
		for j := range spearman[i] {
			if spearman[i][j] >= m.Config.FloatReducerSpearman && m.Model[i] && m.Model[j] {
				m.Model[i] = false
			}
		}
	}
	m.Dimensions = 0
	for i := range m.Model {
		if m.Model[i] {
			m.Dimensions++
		}
	}
}

func (m *FloatReducerModel) GetDimensions() int {
	return m.Dimensions
}

func (m *FloatReducerModel) CalculateFloats(value []float64) []float64 {
	vector := make([]float64, 0)
	for i := range m.Model {
		if m.Model[i] {
			vector = append(vector, value[i])
		}
	}
	return vector
}

func (m *FloatReducerModel) Name() string {
	return "float_reducer"
}

func (m *FloatReducerModel) CalculateString(s string) []float64 {
	return []float64{}
}

func (m *FloatReducerModel) GetQuality() float64 {
	return m.Quality
}

func similarValues(values []float64) bool {
	len := len(values)
	for i, v := range values {
		if i < len-1 {
			if v != values[i+1] {
				return false
			}
		}
	}
	return true
}
