package encoders

import (
	"fmt"
	"testing"
)

func TestFloatReder(t *testing.T) {
	model := "test"
	e := NewEncoder("float reducer test")
	input := NewInput(model, Floats)
	input.AddFloats([]float64{1.0, 0.0, 3.7})
	input.AddFloats([]float64{2.0, 0.9, 0.3})
	input.AddFloats([]float64{3.0, 1.6, 1.3})
	input.AddFloats([]float64{4.0, 2.9, 4.2})
	e.Scan(model, input, FloatReducer)
	e.Transform(model, input)
	e.Explain()
	vector := e.Encode(model, Unified{Float: []float64{1.0, 0.0, 3.7}, Type: Floats})
	if len(vector) != 2 {
		t.Errorf("len: %d != 2", len(vector))
	}
	fmt.Printf("encoded vector: %v", vector)
}
