package encoders

type FloatExactModel struct {
	Dimensions int
	Quality    float64
}

func NewFloatExactModel() *FloatExactModel {
	return &FloatExactModel{}
}

func (m *FloatExactModel) Fit(set *Input) {
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

func (m *FloatExactModel) Name() string {
	return "float_exact"
}

func (m *FloatExactModel) GetQuality() float64 {
	return m.Quality
}
