package net

// NetworkType represents the type of the neural network in terms of
// which task the network has. Classification and Regression are implemented.
type NetworkType int

const (
	// Classification describes the mode of operation: classification
	Classification NetworkType = iota
	// Regression describes the mode of operation: regression
	Regression
)
