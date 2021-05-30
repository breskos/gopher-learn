package encoders

type EncoderConfig struct {
	DelimiterToken    string
	DimToSamplesRatio float64
	// Decision heuristics
	FloatReducerThreshold   int
	TopicModelMinDelimiters int
	NGramsMaxTokens         int
	DictionaryMaxEntries    int
	DictionaryMaxDelimiters int
	// Application settings
	NGramMaxGrams        int
	NGramMaxCapacity     int
	DefaultStringEncoder EncoderType
}

func DefaultConfig() *EncoderConfig {
	return &EncoderConfig{
		DelimiterToken:          " ",
		DimToSamplesRatio:       0.8,
		FloatReducerThreshold:   40,
		TopicModelMinDelimiters: 5,
		NGramsMaxTokens:         20,
		DictionaryMaxEntries:    50,
		NGramMaxGrams:           3,
		DefaultStringEncoder:    StringNGrams,
	}
}
