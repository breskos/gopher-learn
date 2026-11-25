// Package drift detects changes in a stream of numbers, such as the error
// of a model that learns online. A detector receives one value at a time and
// reports when the distribution of the recent values differs from the
// earlier ones.
//
// Feed a detector with the per-sample error of the predictions (0 for
// correct, 1 for wrong, or the absolute error for regression) and react when
// it signals drift, e.g. by forgetting old training data.
package drift

// Detector watches a stream of values.
type Detector interface {
	// Add records a value and reports whether drift was detected.
	Add(x float64) bool
	// Reset forgets all values.
	Reset()
}
