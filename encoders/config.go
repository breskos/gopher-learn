package encoders

// Config holds the heuristics used by Scan and the parameters of the models.
type Config struct {
	// DictionaryMaxEntries is the maximum number of distinct strings for
	// which Scan chooses StringDictionary.
	DictionaryMaxEntries int `json:"dictionary_max_entries"`
	// DictionaryMaxTokens is the maximum number of tokens per string for
	// which Scan chooses StringDictionary.
	DictionaryMaxTokens int `json:"dictionary_max_tokens"`
	// SplitDictionaryMaxEntries is the maximum number of distinct tokens for
	// which Scan chooses StringSplitDictionary.
	SplitDictionaryMaxEntries int `json:"split_dictionary_max_entries"`
	// NGramMaxLength is the maximum string length for which Scan chooses StringNGrams.
	NGramMaxLength int `json:"ngram_max_length"`
	// DefaultStringEncoder is chosen by Scan if no heuristic matches.
	DefaultStringEncoder Type `json:"default_string_encoder"`

	// NGramSize is the number of characters per n-gram.
	NGramSize int `json:"ngram_size"`
	// NGramMaxCapacity is the maximum number of n-grams kept.
	NGramMaxCapacity int `json:"ngram_max_capacity"`
	// NGramCropRatio drops n-grams occurring in a smaller ratio of samples,
	// applied only if the vocabulary exceeds NGramMaxCapacity.
	NGramCropRatio float64 `json:"ngram_crop_ratio"`

	// FloatReducerCorrelation drops a dimension if its absolute Spearman
	// correlation with an already kept dimension is at least this value.
	FloatReducerCorrelation float64 `json:"float_reducer_correlation"`
}

// DefaultConfig returns the default configuration.
func DefaultConfig() Config {
	return Config{
		DictionaryMaxEntries:      50,
		DictionaryMaxTokens:       5,
		SplitDictionaryMaxEntries: 100,
		NGramMaxLength:            20,
		DefaultStringEncoder:      StringNGrams,
		NGramSize:                 3,
		NGramMaxCapacity:          100,
		NGramCropRatio:            0.05,
		FloatReducerCorrelation:   0.9,
	}
}
