package encoders

import (
	"cmp"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
)

// normalize lowercases s and collapses every run of characters that are not
// letters or digits into a single space.
func normalize(s string) string {
	return strings.Join(tokenize(s), " ")
}

func tokenize(s string) []string {
	return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

// chooseStringType picks a string model based on the input data.
func chooseStringType(cfg Config, in *Input) Type {
	uniques := make(map[string]struct{})
	tokens := make(map[string]struct{})
	maxTokens, maxLen := 0, 0
	for _, v := range in.Values {
		norm := normalize(v.String)
		uniques[norm] = struct{}{}
		t := tokenize(v.String)
		for _, tok := range t {
			tokens[tok] = struct{}{}
		}
		maxTokens = max(maxTokens, len(t))
		maxLen = max(maxLen, utf8.RuneCountInString(norm))
	}
	switch {
	case len(uniques) <= cfg.DictionaryMaxEntries && maxTokens <= cfg.DictionaryMaxTokens:
		return StringDictionary
	case len(tokens) < cfg.SplitDictionaryMaxEntries:
		return StringSplitDictionary
	case maxLen <= cfg.NGramMaxLength:
		return StringNGrams
	}
	return cfg.DefaultStringEncoder
}

func requireStrings(in *Input) error {
	if in == nil || len(in.Values) == 0 {
		return errors.New("no values")
	}
	if in.Type != String {
		return errors.New("string model needs string input")
	}
	return nil
}

// DictionaryModel one-hot encodes whole normalized strings.
// Unknown strings encode to the zero vector.
type DictionaryModel struct {
	Entries []string `json:"entries"`

	lookup map[string]int
}

// Fit implements Model.
func (m *DictionaryModel) Fit(in *Input, _ Config) error {
	if err := requireStrings(in); err != nil {
		return err
	}
	m.Entries, m.lookup = nil, nil
	seen := make(map[string]struct{})
	for _, v := range in.Values {
		n := normalize(v.String)
		if _, ok := seen[n]; !ok {
			seen[n] = struct{}{}
			m.Entries = append(m.Entries, n)
		}
	}
	m.lookup = indexOf(m.Entries)
	return nil
}

// Encode implements Model.
func (m *DictionaryModel) Encode(v Value) ([]float64, error) {
	out := make([]float64, len(m.Entries))
	if i, ok := m.lookup[normalize(v.String)]; ok {
		out[i] = 1
	}
	return out, nil
}

// Dimensions implements Model.
func (m *DictionaryModel) Dimensions() int { return len(m.Entries) }

// SplitDictionaryModel multi-hot encodes the tokens of a string.
// Unknown tokens are ignored.
type SplitDictionaryModel struct {
	Tokens []string `json:"tokens"`

	lookup map[string]int
}

// Fit implements Model.
func (m *SplitDictionaryModel) Fit(in *Input, _ Config) error {
	if err := requireStrings(in); err != nil {
		return err
	}
	m.Tokens, m.lookup = nil, nil
	seen := make(map[string]struct{})
	for _, v := range in.Values {
		for _, t := range tokenize(v.String) {
			if _, ok := seen[t]; !ok {
				seen[t] = struct{}{}
				m.Tokens = append(m.Tokens, t)
			}
		}
	}
	m.lookup = indexOf(m.Tokens)
	return nil
}

// Encode implements Model.
func (m *SplitDictionaryModel) Encode(v Value) ([]float64, error) {
	out := make([]float64, len(m.Tokens))
	for _, t := range tokenize(v.String) {
		if i, ok := m.lookup[t]; ok {
			out[i] = 1
		}
	}
	return out, nil
}

// Dimensions implements Model.
func (m *SplitDictionaryModel) Dimensions() int { return len(m.Tokens) }

// NGramModel multi-hot encodes the character n-grams of a string.
type NGramModel struct {
	Size  int      `json:"size"`
	Grams []string `json:"grams"`

	lookup map[string]int
}

// Fit implements Model. If the vocabulary exceeds cfg.NGramMaxCapacity,
// rare n-grams are dropped and only the most frequent ones are kept.
func (m *NGramModel) Fit(in *Input, cfg Config) error {
	if err := requireStrings(in); err != nil {
		return err
	}
	if cfg.NGramSize < 1 {
		return errors.New("NGramSize must be positive")
	}
	m.Size, m.Grams, m.lookup = cfg.NGramSize, nil, nil
	// document frequency: number of samples containing a gram
	df := make(map[string]int)
	for _, v := range in.Values {
		seen := make(map[string]struct{})
		for _, g := range ngrams(normalize(v.String), m.Size) {
			if _, ok := seen[g]; !ok {
				seen[g] = struct{}{}
				df[g]++
			}
		}
	}
	for g := range df {
		m.Grams = append(m.Grams, g)
	}
	slices.SortFunc(m.Grams, func(a, b string) int {
		return cmp.Or(cmp.Compare(df[b], df[a]), cmp.Compare(a, b))
	})
	if cfg.NGramMaxCapacity > 0 && len(m.Grams) > cfg.NGramMaxCapacity {
		minCount := cfg.NGramCropRatio * float64(len(in.Values))
		m.Grams = slices.DeleteFunc(m.Grams, func(g string) bool { return float64(df[g]) < minCount })
		m.Grams = m.Grams[:min(len(m.Grams), cfg.NGramMaxCapacity)]
	}
	m.lookup = indexOf(m.Grams)
	return nil
}

// Encode implements Model.
func (m *NGramModel) Encode(v Value) ([]float64, error) {
	out := make([]float64, len(m.Grams))
	for _, g := range ngrams(normalize(v.String), m.Size) {
		if i, ok := m.lookup[g]; ok {
			out[i] = 1
		}
	}
	return out, nil
}

// Dimensions implements Model.
func (m *NGramModel) Dimensions() int { return len(m.Grams) }

func ngrams(s string, n int) []string {
	r := []rune(s)
	if len(r) < n {
		return nil
	}
	out := make([]string, 0, len(r)-n+1)
	for i := 0; i+n <= len(r); i++ {
		out = append(out, string(r[i:i+n]))
	}
	return out
}

func indexOf(keys []string) map[string]int {
	lookup := make(map[string]int, len(keys))
	for i, k := range keys {
		lookup[k] = i
	}
	return lookup
}

// UnmarshalJSON decodes the model and rebuilds its lookup table.
func (m *DictionaryModel) UnmarshalJSON(b []byte) error {
	type plain DictionaryModel
	if err := json.Unmarshal(b, (*plain)(m)); err != nil {
		return err
	}
	m.lookup = indexOf(m.Entries)
	return nil
}

// UnmarshalJSON decodes the model and rebuilds its lookup table.
func (m *SplitDictionaryModel) UnmarshalJSON(b []byte) error {
	type plain SplitDictionaryModel
	if err := json.Unmarshal(b, (*plain)(m)); err != nil {
		return err
	}
	m.lookup = indexOf(m.Tokens)
	return nil
}

// UnmarshalJSON decodes the model and rebuilds its lookup table.
func (m *NGramModel) UnmarshalJSON(b []byte) error {
	type plain NGramModel
	if err := json.Unmarshal(b, (*plain)(m)); err != nil {
		return err
	}
	m.lookup = indexOf(m.Grams)
	return nil
}
