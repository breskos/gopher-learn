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
	Task nn.Task `json:"task"`
	// Labels holds the class labels in the order of the network outputs.
	Labels []string `json:"labels,omitempty"`
	// Features optionally names the features, e.g. from a CSV header.
	Features []string  `json:"features,omitempty"`
	Samples  []*Sample `json:"samples"`

	index map[uint64]int // number of samples per feature vector key
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
	if err := s.Prepare(sample); err != nil {
		return err
	}
	s.ensureIndex()
	s.index[sample.Key()]++
	s.Samples = append(s.Samples, sample)
	return nil
}

// Replace validates a sample and puts it in place of the sample at index i.
func (s *Set) Replace(i int, sample *Sample) error {
	if i < 0 || i >= len(s.Samples) {
		return fmt.Errorf("learn: index %d out of range for %d samples", i, len(s.Samples))
	}
	if err := s.Prepare(sample); err != nil {
		return err
	}
	s.ensureIndex()
	s.unindex(s.Samples[i])
	s.index[sample.Key()]++
	s.Samples[i] = sample
	return nil
}

// DropOldest removes the first n samples (all if n exceeds the size).
func (s *Set) DropOldest(n int) {
	n = min(max(n, 0), len(s.Samples))
	s.ensureIndex()
	for i := range n {
		s.unindex(s.Samples[i])
		s.Samples[i] = nil
	}
	s.Samples = s.Samples[n:]
}

func (s *Set) unindex(sample *Sample) {
	k := sample.Key()
	if s.index[k] <= 1 {
		delete(s.index, k)
	} else {
		s.index[k]--
	}
}

// Prepare validates a sample for the set and fills in its class index and
// target vector without adding it.
func (s *Set) Prepare(sample *Sample) error {
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
	s.index = make(map[uint64]int, len(s.Samples))
	for _, sm := range s.Samples {
		s.index[sm.Key()]++
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
	sub := &Set{Task: s.Task, Labels: slices.Clone(s.Labels), Features: slices.Clone(s.Features), Samples: make([]*Sample, len(idx))}
	for i, j := range idx {
		sub.Samples[i] = s.Samples[j]
	}
	return sub
}

// Data returns the feature vectors and target vectors of the set for nn.Fit.
// The slices share memory with the samples.
func (s *Set) Data() nn.Data {
	d := nn.Data{X: make([][]float64, len(s.Samples)), Y: make([][]float64, len(s.Samples))}
	for i, sm := range s.Samples {
		d.X[i], d.Y[i] = sm.Vector, sm.Output
	}
	return d
}

// Validate checks the consistency of a set, e.g. after decoding it: a valid
// task, unique labels and samples with finite values, matching sizes and
// correct targets. inputs and outputs, if positive, are the sizes the
// samples must have; otherwise the first sample defines them.
func (s *Set) Validate(inputs, outputs int) error {
	if s.Task != nn.Classification && s.Task != nn.Regression {
		return fmt.Errorf("learn: invalid task %v", s.Task)
	}
	seen := make(map[string]bool, len(s.Labels))
	for _, l := range s.Labels {
		if seen[l] {
			return fmt.Errorf("learn: duplicate label %q", l)
		}
		seen[l] = true
	}
	if outputs <= 0 {
		outputs = s.OutputSize()
	}
	if s.Task == nn.Classification && outputs != len(s.Labels) {
		return fmt.Errorf("learn: %d labels for %d outputs", len(s.Labels), outputs)
	}
	for i, sm := range s.Samples {
		if sm == nil {
			return fmt.Errorf("learn: sample %d is nil", i)
		}
		if inputs <= 0 {
			inputs = len(sm.Vector)
		}
		if len(sm.Vector) != inputs || inputs == 0 {
			return fmt.Errorf("learn: sample %d has %d features, want %d", i, len(sm.Vector), inputs)
		}
		if len(sm.Output) != outputs {
			return fmt.Errorf("learn: sample %d has %d target values, want %d", i, len(sm.Output), outputs)
		}
		for _, v := range append(append(sm.Vector[:len(sm.Vector):len(sm.Vector)], sm.Output...), sm.Value) {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return fmt.Errorf("%w: sample %d", ErrNonFinite, i)
			}
		}
		if s.Task == nn.Classification {
			if sm.Class < 0 || sm.Class >= len(s.Labels) || s.Labels[sm.Class] != sm.Label {
				return fmt.Errorf("learn: sample %d has class %d for label %q", i, sm.Class, sm.Label)
			}
		}
	}
	return nil
}
