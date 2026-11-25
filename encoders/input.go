package encoders

import (
	"fmt"
	"slices"
)

// InputType is the kind of raw input.
type InputType int

const (
	// String input.
	String InputType = iota
	// Floats input (a slice of float64).
	Floats
)

var inputTypeNames = [...]string{"string", "floats"}

// String returns the name of the input type.
func (t InputType) String() string {
	if t < 0 || int(t) >= len(inputTypeNames) {
		return fmt.Sprintf("InputType(%d)", int(t))
	}
	return inputTypeNames[t]
}

// MarshalText implements encoding.TextMarshaler.
func (t InputType) MarshalText() ([]byte, error) {
	if t < 0 || int(t) >= len(inputTypeNames) {
		return nil, fmt.Errorf("encoders: invalid input type %d", int(t))
	}
	return []byte(inputTypeNames[t]), nil
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (t *InputType) UnmarshalText(b []byte) error {
	i := slices.Index(inputTypeNames[:], string(b))
	if i < 0 {
		return fmt.Errorf("encoders: unknown input type %q", b)
	}
	*t = InputType(i)
	return nil
}

// Value is a single raw input value.
type Value struct {
	String string
	Floats []float64
}

// Input is a collection of sample values of one type used for fitting.
type Input struct {
	Type   InputType
	Values []Value
}

// NewInput creates an empty input of the given type.
func NewInput(t InputType) *Input {
	return &Input{Type: t}
}

// AddString appends a string sample.
func (in *Input) AddString(s string) {
	in.Values = append(in.Values, Value{String: s})
}

// AddFloats appends a float slice sample.
func (in *Input) AddFloats(f []float64) {
	in.Values = append(in.Values, Value{Floats: f})
}
