package main

import (
	"fmt"

	"github.com/breskos/gopher-learn/encoders"
)

const (
	model = "floatreducer"
)

func main() {
	e := encoders.NewEncoder("float reducer test")
	input := encoders.NewInput(model, encoders.Floats)
	// here we create some data points.
	// As you can see dimension 0 and 1 are very much correlated.
	// From these data points the float reducer strips one of the dimensions of 0 and 1.
	// This FloatReducer is very interesting if you have large float spaces.
	// It also erases dimensions that are not necessary because they just deliver one value or no information gain.
	input.AddFloats([]float64{1.0, 0.0, 3.7})
	input.AddFloats([]float64{2.0, 0.9, 0.3})
	input.AddFloats([]float64{3.0, 1.6, 1.3})
	input.AddFloats([]float64{4.0, 2.9, 4.2})
	e.Scan(model, input, encoders.FloatReducer)
	e.Transform(model, input)
	e.Explain()
	vector := e.Encode(model, encoders.Unified{Float: []float64{1.0, 0.0, 3.7}, Type: encoders.Floats})
	fmt.Printf("encoded vector: %v", vector)

}
