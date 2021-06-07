package encoders

import "encoding/json"

type FloatExactModel struct {
	Dimensions int
	Quality    float64
}

func NewFloatExactModel() *FloatExactModel {
	return &FloatExactModel{}
}

func (m *FloatExactModel) Fit(set *Input, config *EncoderConfig) {
	m.Dimensions = len(set.Values[0].Float)
}

func (m *FloatExactModel) CalculateString(s string) []float64 {
	return make([]float64, m.Dimensions)
}

func (m *FloatExactModel) GetDimensions() int {
	return m.Dimensions
}

func (m *FloatExactModel) CalculateFloats(value []float64) []float64 {
	return value
}

func (m *FloatExactModel) ToDump() ([]byte, error) {
	return json.Marshal(m)
}

func (m *FloatExactModel) FromDump(dump []byte) error {
	return json.Unmarshal(dump, m)
}

func (m *FloatExactModel) Name() string {
	return "float_exact"
}

func (m *FloatExactModel) GetQuality() float64 {
	return m.Quality
}
