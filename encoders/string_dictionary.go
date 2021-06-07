package encoders

import (
	"encoding/json"
	"fmt"
)

type DictionaryModel struct {
	Dimensions int
	Dictionary []string
	Quality    float64
}

func NewDictionaryModel() *DictionaryModel {
	return &DictionaryModel{}
}

func (m *DictionaryModel) Fit(set *Input, config *EncoderConfig) {
	for _, sample := range set.Values {
		value := normalizeString(sample.String)
		fmt.Printf("%s", value)
		if !contains(m.Dictionary, value) {
			m.Dictionary = append(m.Dictionary, value)
		}
	}
	fmt.Printf("%v", m.Dictionary)
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

func (m *DictionaryModel) Name() string {
	return "dictionary"
}

func (m *DictionaryModel) GetQuality() float64 {
	return m.Quality
}

func (m *DictionaryModel) ToDump() ([]byte, error) {
	return json.Marshal(m)
}

func (m *DictionaryModel) FromDump(dump []byte) error {
	return json.Unmarshal(dump, m)
}

func getIndex(s []string, value string) int {
	for k, v := range s {
		if v == value {
			return k
		}
	}
	return -1
}
