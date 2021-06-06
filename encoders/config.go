package encoders

type EncoderConfig struct {
	DelimiterToken    string
	DimToSamplesRatio float64
	// Decision heuristics
	FloatReducerThreshold     int
	TopicModelMinDelimiters   int
	NGramsMaxTokens           int
	DictionaryMaxEntries      int
	DictionaryMaxDelimiters   int
	SplitDictionaryMaxEntries int
	// Application settings
	FloatReducerSpearman   float64
	FloatReducerSkewness   float64
	FloatReducerZeroValues bool
	NGramMaxGrams          int
	NGramMaxCapacity       int
	NGramCropRatio         float64
	DefaultStringEncoder   EncoderType
}

func DefaultConfig() *EncoderConfig {
	return &EncoderConfig{
		DelimiterToken:            " ",
		DimToSamplesRatio:         0.8,
		FloatReducerThreshold:     40,
		TopicModelMinDelimiters:   5,
		NGramsMaxTokens:           20,
		DictionaryMaxEntries:      50,
		DictionaryMaxDelimiters:   5,
		SplitDictionaryMaxEntries: 100,
		FloatReducerSpearman:      0.90,
		FloatReducerSkewness:      0.90,
		FloatReducerZeroValues:    true,
		NGramMaxGrams:             3,
		NGramMaxCapacity:          100,
		NGramCropRatio:            0.05,
		DefaultStringEncoder:      StringNGrams,
	}
}
