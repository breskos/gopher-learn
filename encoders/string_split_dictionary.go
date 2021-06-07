package encoders

import (
	"encoding/json"
	"fmt"
	"strings"
)

const (
	splitDictionaryDelimiter = " "
)

type SplitDictionaryModel struct {
	Dimensions int
	Dictionary []string
	Quality    float64
}

func NewSplitDictionaryModel() *SplitDictionaryModel {
	return &SplitDictionaryModel{}
}

func (m *SplitDictionaryModel) Fit(set *Input, config *EncoderConfig) {
	delimiter := config.DelimiterToken
	for _, sample := range set.Values {
		value := normalizeString(sample.String)
		fmt.Printf("%s", value)
		values := strings.Split(value, delimiter)
		for _, v := range values {
			if !contains(m.Dictionary, v) {
				m.Dictionary = append(m.Dictionary, v)
			}
		}
	}
	fmt.Printf("%v", m.Dictionary)
	m.Dimensions = len(m.Dictionary)
}

func (m *SplitDictionaryModel) CalculateString(s string) []float64 {
	vector := make([]float64, m.Dimensions)
	idx := getIndex(m.Dictionary, s)
	if idx != -1 {
		vector[idx] = 1.0
	}
	return vector
}

func (m *SplitDictionaryModel) GetDimensions() int {
	return m.Dimensions
}

func (m *SplitDictionaryModel) CalculateFloats([]float64) []float64 {
	return []float64{}
}

func (m *SplitDictionaryModel) ToDump() ([]byte, error) {
	return json.Marshal(m)
}

func (m *SplitDictionaryModel) FromDump(dump []byte) error {
	return json.Unmarshal(dump, m)
}

func (m *SplitDictionaryModel) Name() string {
	return "splitted_dictionary"
}

func (m *SplitDictionaryModel) GetQuality() float64 {
	return m.Quality
}
