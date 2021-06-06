package encoders

import (
	"sort"
)

const (
	DefaultGram = 3
)

type NGramModel struct {
	Dimensions int
	// Grams to index in vector
	GramsLookup map[string]int
	// Grams to number of appearances
	Grams       map[string]int
	Samples     int
	MaxGrams    int
	MaxCapacity int
	CropRatio   float64
	Quality     float64
}

func NewNGramModel(config *EncoderConfig) *NGramModel {
	return &NGramModel{
		Grams:       make(map[string]int, 0),
		GramsLookup: make(map[string]int),
		MaxGrams:    config.NGramMaxGrams,
		MaxCapacity: config.NGramMaxCapacity,
		CropRatio:   config.NGramCropRatio,
	}
}

func (m *NGramModel) Fit(set *Input) {
	modelIndex := 0
	for _, sample := range set.Values {
		m.Samples++
		value := normalizeString(sample.String)
		l := len(value)
		for k := range value {
			if k <= l-DefaultGram {
				gram := value[k : k+DefaultGram]
				if _, ok := m.GramsLookup[gram]; !ok {
					m.GramsLookup[gram] = modelIndex
					modelIndex++
					m.Grams[gram] = 1
				} else {
					m.Grams[gram]++
				}
			}
		}
	}
	m.Dimensions = len(m.Grams)
	m.optimize()
}

func (m *NGramModel) CalculateString(s string) []float64 {
	vector := make([]float64, m.Dimensions)
	value := normalizeString(s)
	ngrams := ngramize(value, DefaultGram)
	for _, gram := range ngrams {
		if index, ok := m.GramsLookup[gram]; ok {
			vector[index] = 1.0
		}
	}
	return vector
}

func (m *NGramModel) GetDimensions() int {
	return m.Dimensions
}

func (m *NGramModel) CalculateFloats([]float64) []float64 {
	return []float64{}
}

func (m *NGramModel) Name() string {
	return "ngrams"
}

func (m *NGramModel) GetQuality() float64 {
	return m.Quality
}

func (m *NGramModel) optimize() {
	if m.MaxCapacity >= m.Dimensions {
		return
	}

	for gram, appearance := range m.Grams {
		if float64(appearance)/float64(m.Samples) < m.CropRatio {
			delete(m.Grams, gram)
		}
	}
	// reindex cropped to vector index
	m.GramsLookup = make(map[string]int)
	index := 0
	for gram := range m.Grams {
		m.GramsLookup[gram] = index
		index++
	}
	m.Dimensions = len(m.Grams)
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
	sorted := make(map[string]int)
	for _, kv := range ps {
		sorted[kv.Key] = kv.Value
	}
	return sorted
}

func ngramize(value string, n int) []string {
	l := len(value)
	grams := make([]string, 0)
	for k := range value {
		if k <= l-n {
			gram := value[k : k+n]
			grams = append(grams, gram)
		}
	}
	return grams
}
