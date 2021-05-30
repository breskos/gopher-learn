package encoders

import (
	"math"
	"strings"
)

// The scanner functions are used to determine whether a set is likely to fit and how many dimensions are
// suitable for this set of data. Here also the decision is made for a string encoder.

// evaluates if the size of the given sample set is sufficient for the autoencoder.
// If you just apply float values in the input vector this should be taken for granted.
// For strings it tells a different story, since one string (regardless how long) is one dimension.
func evalDimToSamples(dims int, samples int) float64 {
	if samples < dims {
		return 0.0
	}
	return math.Min(float64(samples)*10.0/float64(dims), 1.0)
}

// Here the decision for an string encoder is made based on the provided configuration file.
func evaluateStrings(config *EncoderConfig, values []string) EncoderType {
	maxDelimiters := 0
	maxStringLength := 0
	samples := 0
	uniques := make([]string, 0)
	for _, v := range values {
		l := len(strings.Split(v, config.DelimiterToken))
		if l > maxDelimiters {
			maxDelimiters = l
		}
		l = len(v)
		if l > maxStringLength {
			maxStringLength = l
		}
		samples++
		if !contains(uniques, v) {
			uniques = append(uniques, v)
		}
	}
	uniqueEntries := len(uniques)
	// Dictionary makes sense if you have something like string states of something.
	// In this case the dictionary assignes 0,1 for each dictionary entry shown.
	if uniqueEntries <= config.DictionaryMaxEntries && maxDelimiters <= config.DictionaryMaxDelimiters {
		return StringDictionary
	}
	// If there are too much entries to use it as a dictionary, we try to make NGrams out of it.
	if maxStringLength <= config.NGramsMaxTokens {
		return StringNGrams
	}
	// If NGrams dont match we try to apply topic modelling.
	if maxDelimiters >= config.TopicModelMinDelimiters {
		return StringTopics
	}
	// If nothing matches we default to NGrams.
	return config.DefaultStringEncoder
}

func contains(s []string, str string) bool {
	for _, v := range s {
		if v == str {
			return true
		}
	}
	return false
}
