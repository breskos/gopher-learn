package encoders

type DictionaryModel struct {
	Dimensions int
	Dictionary []string
}

func NewDictionaryModel() *DictionaryModel {
	return &DictionaryModel{}
}

func (m *DictionaryModel) Fit(dimensions []int, set *Set) {
	for _, dim := range dimensions {
		for _, sample := range set.Samples {
			value := sample.Vector[dim].String
			if !contains(m.Dictionary, value) {
				m.Dictionary = append(m.Dictionary, value)
			}
		}
	}
	m.Dimensions = len(m.Dictionary)
}

func (m *DictionaryModel) CalculateString(s string) []float64 {
	vector := make([]float64, m.Dimensions)
	idx := getIndex(m.Dictionary, s)
	if idx != -1 {
		vector[idx] = 1.0
	}
	return vector
}

func (m *DictionaryModel) GetDimensions() int {
	return m.Dimensions
}

func (m *DictionaryModel) CalculateFloats([]float64) []float64 {
	return []float64{}
}

func (m *DictionaryModel) Quality() float64 {
	return 1.0
}

func (m *DictionaryModel) Name() string {
	return "dictionary"
}

func getIndex(s []string, value string) int {
	for k, v := range s {
		if v == value {
			return k
		}
	}
	return -1
}
