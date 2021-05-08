package neural

type NetworkType int

const (
	// Classification describes the mode of operation: classification
	Classification NetworkType = iota
	// Regression describes the mode of operation: regression
	Regression
)
