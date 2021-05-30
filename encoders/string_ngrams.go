package encoders

import (
	"sort"
	"strings"
)

const (
	DefaultGram = 3
)

type NGramModel struct {
	Dimensions  int
	Grams       []string
	MaxGrams    int
	MaxCapacity int
}

func NewNGramModel(config *EncoderConfig) *NGramModel {
	return &NGramModel{
		Grams:       make([]string, 0),
		MaxGrams:    config.NGramMaxGrams,
		MaxCapacity: config.NGramMaxCapacity,
	}
}

func (m *NGramModel) Fit(dimensions []int, set *Set) {
	var appearances map[string]int
	for _, dim := range dimensions {
		for _, sample := range set.Samples {
			value := sample.Vector[dim].String
			value = strings.ToLower(value)
			l := len(value)
			for k := range value {
				if k <= l-DefaultGram {
					gram := value[k : k+DefaultGram]
					if _, ok := appearances[gram]; !ok {
						appearances[gram] = 1
					} else {
						appearances[gram]++
					}
				}
			}
		}
	}
	m.set(appearances)
}

func (m *NGramModel) CalculateString(s string) []float64 {
	return []float64{}
}

func (m *NGramModel) GetDimensions() int {
	return m.Dimensions
}

func (m *NGramModel) CalculateFloats([]float64) []float64 {
	return []float64{}
}

func (m *NGramModel) Quality() float64 {
	return 1.0
}

func (m *NGramModel) Name() string {
	return "ngrams"
}

func (m *NGramModel) set(appearances map[string]int) {
	l := len(appearances)
	if m.MaxCapacity > l {
		for gram := range appearances {
			m.Grams = append(m.Grams, gram)
		}
		m.Dimensions = l
		return
	}
	// if max capacity is below the apperances we trim down
	appearances = sortByValue(appearances)
	i := 0
	for gram := range appearances {
		if i < m.MaxCapacity {
			m.Grams = append(m.Grams, gram)
		} else {
			m.Dimensions = len(m.Grams)
			return
		}
	}

}

func sortByValue(m map[string]int) map[string]int {
	type pair struct {
		Key   string
		Value int
	}

	var ps []pair
	for k, v := range m {
		ps = append(ps, pair{k, v})
	}

	sort.Slice(ps, func(i, j int) bool {
		return ps[i].Value > ps[j].Value
	})

	var sorted map[string]int
	for _, kv := range ps {
		sorted[kv.Key] = kv.Value
	}
	return sorted
}
