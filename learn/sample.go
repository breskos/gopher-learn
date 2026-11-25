// Package learn holds labeled data sets used to train and evaluate networks.
package learn

import (
	"encoding/binary"
	"hash/fnv"
	"math"
)

// Sample is a single data point.
//
// For classification, Label names the class; Class and Output (one-hot) are
// filled in when the sample is added to a Set. For regression, Value is the
// target and Output is set to [Value].
type Sample struct {
	Vector []float64 `json:"vector"`
	Output []float64 `json:"output"`
	Label  string    `json:"label,omitempty"`
	Class  int       `json:"class"`
	Value  float64   `json:"value,omitempty"`
}

// NewClassificationSample creates a sample for classification.
func NewClassificationSample(vector []float64, label string) *Sample {
	return &Sample{Vector: vector, Label: label}
}

// NewRegressionSample creates a sample for regression.
func NewRegressionSample(vector []float64, value float64) *Sample {
	return &Sample{Vector: vector, Value: value}
}

// Key returns a hash of the feature vector, used to detect duplicates.
func (s *Sample) Key() uint64 {
	h := fnv.New64a()
	var buf [8]byte
	for _, v := range s.Vector {
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(v))
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}
