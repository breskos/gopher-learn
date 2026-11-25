// Package encoders turns raw inputs (strings and float slices) into feature
// vectors for neural networks.
//
// An Encoder holds one fitted Model per named input dimension. Scan decides
// which model to use (automatically or explicitly), Fit learns the model from
// sample data and Encode converts new values.
package encoders

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
)

// ErrUnknownDimension is returned for a dimension name the encoder does not know.
var ErrUnknownDimension = errors.New("encoders: unknown dimension")

// Type selects an encoding model.
type Type int

const (
	// Automatic lets Scan pick a model based on heuristics.
	Automatic Type = iota
	// StringDictionary one-hot encodes whole (normalized) strings.
	StringDictionary
	// StringSplitDictionary multi-hot encodes the tokens of a string.
	StringSplitDictionary
	// StringNGrams multi-hot encodes the character n-grams of a string.
	StringNGrams
	// FloatExact passes float slices through unchanged.
	FloatExact
	// FloatReducer drops constant and highly correlated float dimensions.
	FloatReducer
	// StringHashing hashes the tokens of a string into a fixed number of
	// dimensions (the hashing trick). It needs no vocabulary, so new words
	// in a stream are encoded too.
	StringHashing
	// StringTFIDF weights the tokens of a string by how rare they are in the
	// fitted texts (term frequency times inverse document frequency).
	StringTFIDF
	// FloatCategories one-hot encodes numbers that are category codes, such
	// as 1 = red, 2 = green, per dimension.
	FloatCategories
)

var typeNames = [...]string{"automatic", "string-dictionary", "string-split-dictionary", "string-ngrams", "float-exact", "float-reducer",
	"string-hashing", "string-tfidf", "float-categories"}

// String returns the name of the type.
func (t Type) String() string {
	if t < 0 || int(t) >= len(typeNames) {
		return fmt.Sprintf("Type(%d)", int(t))
	}
	return typeNames[t]
}

// MarshalText implements encoding.TextMarshaler.
func (t Type) MarshalText() ([]byte, error) {
	if t < 0 || int(t) >= len(typeNames) {
		return nil, fmt.Errorf("encoders: invalid type %d", int(t))
	}
	return []byte(typeNames[t]), nil
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (t *Type) UnmarshalText(b []byte) error {
	i := slices.Index(typeNames[:], string(b))
	if i < 0 {
		return fmt.Errorf("encoders: unknown type %q", b)
	}
	*t = Type(i)
	return nil
}

func (t Type) inputType() InputType {
	if t == FloatExact || t == FloatReducer || t == FloatCategories {
		return Floats
	}
	return String
}

// Model is a fitted encoding for one input dimension.
type Model interface {
	// Fit learns the model from sample data.
	Fit(in *Input, cfg Config) error
	// Encode converts a value into a feature vector of length Dimensions.
	Encode(v Value) ([]float64, error)
	// Dimensions returns the length of the produced feature vectors.
	Dimensions() int
}

// NewModel returns an unfitted model of the given type.
func NewModel(t Type) (Model, error) {
	switch t {
	case StringDictionary:
		return &DictionaryModel{}, nil
	case StringSplitDictionary:
		return &SplitDictionaryModel{}, nil
	case StringNGrams:
		return &NGramModel{}, nil
	case FloatExact:
		return &FloatExactModel{}, nil
	case FloatReducer:
		return &FloatReducerModel{}, nil
	case StringHashing:
		return &HashingModel{}, nil
	case StringTFIDF:
		return &TFIDFModel{}, nil
	case FloatCategories:
		return &FloatCategoriesModel{}, nil
	}
	return nil, fmt.Errorf("encoders: no model for type %v", t)
}

// Dimension is the encoding of one named input.
type Dimension struct {
	InputType InputType `json:"input_type"`
	Type      Type      `json:"type"`
	// Model is nil until the dimension is fitted.
	Model Model `json:"model,omitempty"`
}

// UnmarshalJSON decodes the dimension including its concrete model.
func (d *Dimension) UnmarshalJSON(b []byte) error {
	var raw struct {
		InputType InputType       `json:"input_type"`
		Type      Type            `json:"type"`
		Model     json.RawMessage `json:"model"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	d.InputType, d.Type, d.Model = raw.InputType, raw.Type, nil
	if len(raw.Model) == 0 || string(raw.Model) == "null" {
		return nil
	}
	m, err := NewModel(raw.Type)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw.Model, m); err != nil {
		return err
	}
	d.Model = m
	return nil
}

// Encoder encodes several named inputs.
type Encoder struct {
	Name       string                `json:"name"`
	Config     Config                `json:"config"`
	Dimensions map[string]*Dimension `json:"dimensions"`
}

// New creates an encoder with the default configuration.
func New(name string) *Encoder {
	return &Encoder{Name: name, Config: DefaultConfig(), Dimensions: make(map[string]*Dimension)}
}

// Scan chooses the model type for a dimension. With Automatic the type is
// derived from the input data.
func (e *Encoder) Scan(name string, in *Input, t Type) error {
	if in == nil || len(in.Values) == 0 {
		return errors.New("encoders: no input values to scan")
	}
	if e.Dimensions == nil {
		e.Dimensions = make(map[string]*Dimension)
	}
	if t == Automatic {
		if in.Type == Floats {
			t = FloatExact
		} else {
			t = chooseStringType(e.Config, in)
		}
	}
	if t.inputType() != in.Type {
		return fmt.Errorf("encoders: type %v cannot encode %v input", t, in.Type)
	}
	e.Dimensions[name] = &Dimension{InputType: in.Type, Type: t}
	return nil
}

// Fit learns the model of a dimension. If the dimension was not scanned
// before, it is scanned automatically.
func (e *Encoder) Fit(name string, in *Input) error {
	d, ok := e.Dimensions[name]
	if !ok {
		if err := e.Scan(name, in, Automatic); err != nil {
			return err
		}
		d = e.Dimensions[name]
	}
	if in.Type != d.InputType {
		return fmt.Errorf("encoders: dimension %q expects %v input, got %v", name, d.InputType, in.Type)
	}
	m, err := NewModel(d.Type)
	if err != nil {
		return err
	}
	if err := m.Fit(in, e.Config); err != nil {
		return fmt.Errorf("encoders: fit %q: %w", name, err)
	}
	d.Model = m
	return nil
}

// Encode converts a value of a fitted dimension into a feature vector.
func (e *Encoder) Encode(name string, v Value) ([]float64, error) {
	d, ok := e.Dimensions[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownDimension, name)
	}
	if d.Model == nil {
		return nil, fmt.Errorf("encoders: dimension %q is not fitted", name)
	}
	return d.Model.Encode(v)
}

// EncodeString is a shortcut for Encode with a string value.
func (e *Encoder) EncodeString(name, s string) ([]float64, error) {
	return e.Encode(name, Value{String: s})
}

// EncodeFloats is a shortcut for Encode with a float slice value.
func (e *Encoder) EncodeFloats(name string, f []float64) ([]float64, error) {
	return e.Encode(name, Value{Floats: f})
}

// Explain writes which model each dimension uses.
func (e *Encoder) Explain(w io.Writer) error {
	names := make([]string, 0, len(e.Dimensions))
	for n := range e.Dimensions {
		names = append(names, n)
	}
	slices.Sort(names)
	var b strings.Builder
	for _, n := range names {
		d := e.Dimensions[n]
		dims := 0
		if d.Model != nil {
			dims = d.Model.Dimensions()
		}
		fmt.Fprintf(&b, "[%s] %v input, %v, %d dimensions\n", n, d.InputType, d.Type, dims)
	}
	_, err := io.WriteString(w, b.String())
	return err
}
