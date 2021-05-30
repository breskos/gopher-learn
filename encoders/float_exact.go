package encoders

type FloatExactModel struct {
}

func NewFloatExactModel() *FloatExactModel {
	return &FloatExactModel{}
}

func (m *FloatExactModel) Fit(dimensions []int, set *Set) {
	// this is a straight forward operation from the float_exact model
}

func (m *FloatExactModel) CalculateString(s string) []float64 {
	return []float64{}
}

func (m *FloatExactModel) GetDimensions() int {
	return 1
}

func (m *FloatExactModel) CalculateFloats(value []float64) []float64 {
	return value
}

func (m *FloatExactModel) Quality() float64 {
	return 1.0
}

func (m *FloatExactModel) Name() string {
	return "float_exact"
}
