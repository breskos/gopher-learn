package encoders

type StringEncoder int

const (
	// StringExact uses exact matches on strings as dictionary approach
	StringExact StringEncoder = iota
	// StringTopics uses topic modelling on strings
	StringTopics
	// StringNGrams uses N-Gram modelling on strings
	StringNGrams
)
