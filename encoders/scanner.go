package encoders

import (
	"strings"
)

// The scanner functions are used to determine whether a set is likely to fit and how many dimensions are
// suitable for this set of data. Here also the decision is made for a string encoder.

// Here the decision for an string encoder is made based on the provided configuration file.
func evaluateStrings(config *EncoderConfig, values *Input) EncoderType {
	maxDelimiters := 0
	maxStringLength := 0
	samples := 0
	uniques := make([]string, 0)
	uniqueTokens := make(map[string]int)
	for _, v := range values.Values {
		tokens := strings.Split(v.String, config.DelimiterToken)
		for _, v := range tokens {
			if _, ok := uniqueTokens[v]; ok {
				uniqueTokens[v]++
			} else {
				uniqueTokens[v] = 1
			}
		}
		l := len(tokens)
		if l > maxDelimiters {
			maxDelimiters = l
		}
		l = len(v.String)
		if l > maxStringLength {
			maxStringLength = l
		}
		samples++
		if !contains(uniques, v.String) {
			uniques = append(uniques, v.String)
		}
	}
	uniqueEntries := len(uniques)
	// Dictionary makes sense if you have something like string states of something.
	// In this case the dictionary assignes 0,1 for each dictionary entry shown.
	if uniqueEntries <= config.DictionaryMaxEntries && maxDelimiters <= config.DictionaryMaxDelimiters {
		return StringDictionary
	}
	// SplittedDictionary make sense if there are not that much symbols but not just one token.
	// Also sentences in a small space are possible.
	if len(uniqueTokens) < config.SplitDictionaryMaxEntries {
		return StringSplitDictionary
	}
	// If there are too much entries to use it as a dictionary, we try to make NGrams out of it.
	if maxStringLength <= config.NGramsMaxTokens {
		return StringNGrams
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
