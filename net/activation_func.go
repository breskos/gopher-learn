package net

import "math"

// ActivationFunction function definition of activation functions
type ActivationFunction func(float64) float64

// NewLogisticFunc applies and returns the Logistic function
func NewLogisticFunc(a float64) ActivationFunction {
	return func(x float64) float64 {
		return LogisticFunc(x, a)
	}
}

// LogisticFunc returns the value of the logistic function for x and a
func LogisticFunc(x, a float64) float64 {
	return 1 / (1 + math.Exp(-a*x))
}
