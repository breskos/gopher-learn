package learn

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"

	"github.com/breskos/gopher-learn/nn"
)

var (
	// ErrUnknownClass is returned when a sample references a class the set does not know.
	ErrUnknownClass = errors.New("learn: unknown class")
	// ErrTaskMismatch is returned for operations that do not fit the task of the set.
	ErrTaskMismatch = errors.New("learn: operation does not match task")
	// ErrNonFinite is returned for samples containing NaN or infinite values.
	ErrNonFinite = errors.New("learn: non-finite value")
)

// Set is a collection of samples for one task.
type Set struct {
	Task    nn.Task   `json:"task"`
	Labels  []string  `json:"labels,omitempty"`
	Samples []*Sample `json:"samples"`

	index map[uint64]struct{}
}

// NewSet creates an empty set.
func NewSet(task nn.Task) *Set {
	return &Set{Task: task}
}

// Len returns the number of samples.
func (s *Set) Len() int { return len(s.Samples) }

// InputSize returns the length of the feature vectors, or 0 for an empty set.
func (s *Set) InputSize() int {
	if len(s.Samples) == 0 {
		return 0
	}
	return len(s.Samples[0].Vector)
}

// OutputSize returns the length of the output vectors.
func (s *Set) OutputSize() int {
	if s.Task == nn.Regression {
		return 1
	}
	return len(s.Labels)
}

// AddClass registers a new class label. Classes are indexed in the order
// they are added, which defines the order of the network outputs. The target
// vectors of existing samples are extended accordingly.
func (s *Set) AddClass(label string) error {
	if s.Task != nn.Classification {
		return fmt.Errorf("%w: AddClass on %v set", ErrTaskMismatch, s.Task)
	}
	if slices.Contains(s.Labels, label) {
		return fmt.Errorf("learn: class %q already exists", label)
	}
	s.Labels = append(s.Labels, label)
	for _, sm := range s.Samples {
		sm.Output = append(sm.Output, 0)
	}
	return nil
}

// ClassIndex returns the index of a class label.
func (s *Set) ClassIndex(label string) (int, bool) {
	i := slices.Index(s.Labels, label)
	return i, i >= 0
}

// OneHot returns the target vector of a class label.
func (s *Set) OneHot(label string) ([]float64, error) {
	i, ok := s.ClassIndex(label)
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownClass, label)
	}
	v := make([]float64, len(s.Labels))
	v[i] = 1
	return v, nil
}

// Add validates a sample, fills in its target vector and appends it.
func (s *Set) Add(sample *Sample) error {
	if sample == nil {
		return errors.New("learn: nil sample")
	}
	if len(sample.Vector) == 0 {
		return errors.New("learn: sample has no features")
	}
	if n := s.InputSize(); n > 0 && len(sample.Vector) != n {
		return fmt.Errorf("learn: sample has %d features, set has %d", len(sample.Vector), n)
	}
	for i, v := range sample.Vector {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("%w: feature %d is %v", ErrNonFinite, i, v)
		}
	}
	if math.IsNaN(sample.Value) || math.IsInf(sample.Value, 0) {
		return fmt.Errorf("%w: target value is %v", ErrNonFinite, sample.Value)
	}
	switch s.Task {
	case nn.Classification:
		out, err := s.OneHot(sample.Label)
		if err != nil {
			return err
		}
		sample.Class = slices.Index(out, 1)
		sample.Output = out
	case nn.Regression:
		sample.Output = []float64{sample.Value}
	default:
		return fmt.Errorf("learn: invalid task %v", s.Task)
	}
	s.ensureIndex()
	s.index[sample.Key()] = struct{}{}
	s.Samples = append(s.Samples, sample)
	return nil
}

// Contains reports whether a sample with the same feature vector exists.
func (s *Set) Contains(sample *Sample) bool {
	s.ensureIndex()
	_, ok := s.index[sample.Key()]
	return ok
}

func (s *Set) ensureIndex() {
	if s.index != nil {
		return
	}
	s.index = make(map[uint64]struct{}, len(s.Samples))
	for _, sm := range s.Samples {
		s.index[sm.Key()] = struct{}{}
	}
}

// Distribution returns the number of samples per class label.
func (s *Set) Distribution() map[string]int {
	d := make(map[string]int, len(s.Labels))
	for _, sm := range s.Samples {
		d[sm.Label]++
	}
	return d
}

// Split shuffles the samples and divides them into two sets, the first one
// holding the given ratio of samples. Both sets share the sample pointers.
// If the set has at least two samples, neither part is empty.
func (s *Set) Split(ratio float64, rng *rand.Rand) (*Set, *Set, error) {
	if ratio <= 0 || ratio >= 1 {
		return nil, nil, fmt.Errorf("learn: split ratio must be in (0, 1), got %v", ratio)
	}
	if rng == nil {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	idx := rng.Perm(len(s.Samples))
	cut := int(math.Round(ratio * float64(len(idx))))
	if len(idx) >= 2 {
		cut = min(max(cut, 1), len(idx)-1)
	}
	a, b := s.subset(idx[:cut]), s.subset(idx[cut:])
	return a, b, nil
}

func (s *Set) subset(idx []int) *Set {
	sub := &Set{Task: s.Task, Labels: slices.Clone(s.Labels), Samples: make([]*Sample, len(idx))}
	for i, j := range idx {
		sub.Samples[i] = s.Samples[j]
	}
	return sub
}
