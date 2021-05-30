package encoders

import neural "github.com/breskos/gopher-learn"

type SampleType int

const (
	ValueString SampleType = iota
	ValueFloat
)

// Sample holds the sample data, value is just used for regression annotation
type Sample struct {
	Vector      []*MultiType
	Output      []float64
	Value       float64 // used for regression
	Label       string  // used for classification
	ClassNumber int
	Type        neural.NetworkType
}

// Multitype represents the type we found in the sample file
type MultiType struct {
	String string
	Float  float64
	Type   SampleType
}

// NewClassificationSample creates a new sample data point for classification
func NewClassificationSample(vector []*MultiType, classLabel string) *Sample {
	return &Sample{
		Vector: vector,
		Output: make([]float64, 0),
		Label:  classLabel,
	}
}

// Sample creates a new sample data point for regression
func NewRegressionSample(vector []*MultiType, value float64) *Sample {
	return &Sample{
		Vector: vector,
		Output: make([]float64, 0),
		Value:  value,
	}
}

func NewSample(t neural.NetworkType) *Sample {
	return &Sample{
		Vector: make([]*MultiType, 0),
		Type:   t,
	}
}

func (s *Sample) SetClassLabel(c string) {
	s.Label = c
}

func (s *Sample) SetRegressionTarget(t float64) {
	s.Value = t
}

func (s *Sample) AddDimension(t SampleType, str string, flo float64) {
	s.Vector = append(s.Vector, &MultiType{Type: t, String: str, Float: flo})
}
