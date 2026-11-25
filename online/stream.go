package online

import (
	"fmt"
	"slices"

	"github.com/breskos/gopher-learn/drift"
	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

// Memory selects how the learner bounds its stored samples.
type Memory int

const (
	// Unbounded keeps every sample.
	Unbounded Memory = iota
	// Window keeps the MemorySize most recent samples.
	Window
	// Reservoir keeps a uniform random sample of MemorySize samples from
	// the whole stream (reservoir sampling), so old concepts stay
	// represented until a drift is detected.
	Reservoir
)

// Detector selects the drift detector.
type Detector int

const (
	// NoDrift turns drift detection off.
	NoDrift Detector = iota
	// ADWIN uses drift.ADWIN, which also estimates how many recent samples
	// belong to the new concept.
	ADWIN
	// PageHinkley uses drift.PageHinkley, which detects rising errors fast.
	PageHinkley
)

var (
	memoryNames   = [...]string{"unbounded", "window", "reservoir"}
	detectorNames = [...]string{"none", "adwin", "page-hinkley"}
)

func name(names []string, i int, kind string) string {
	if i < 0 || i >= len(names) {
		return fmt.Sprintf("%s(%d)", kind, i)
	}
	return names[i]
}

func marshal(names []string, i int, kind string) ([]byte, error) {
	if i < 0 || i >= len(names) {
		return nil, fmt.Errorf("online: invalid %s %d", kind, i)
	}
	return []byte(names[i]), nil
}

func unmarshal(names []string, b []byte, kind string) (int, error) {
	i := slices.Index(names, string(b))
	if i < 0 {
		return 0, fmt.Errorf("online: unknown %s %q", kind, b)
	}
	return i, nil
}

// String returns the name of the memory.
func (m Memory) String() string { return name(memoryNames[:], int(m), "Memory") }

// MarshalText implements encoding.TextMarshaler.
func (m Memory) MarshalText() ([]byte, error) { return marshal(memoryNames[:], int(m), "memory") }

// UnmarshalText implements encoding.TextUnmarshaler.
func (m *Memory) UnmarshalText(b []byte) error {
	i, err := unmarshal(memoryNames[:], b, "memory")
	*m = Memory(i)
	return err
}

// String returns the name of the detector.
func (d Detector) String() string { return name(detectorNames[:], int(d), "Detector") }

// MarshalText implements encoding.TextMarshaler.
func (d Detector) MarshalText() ([]byte, error) {
	return marshal(detectorNames[:], int(d), "drift detector")
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (d *Detector) UnmarshalText(b []byte) error {
	i, err := unmarshal(detectorNames[:], b, "drift detector")
	*d = Detector(i)
	return err
}

// DriftEvent describes a detected drift.
type DriftEvent struct {
	// Sample is the number of samples Learn had processed, including the
	// one that triggered the detection.
	Sample int
	// Kept and Forgotten are the numbers of stored samples kept and dropped.
	Kept, Forgotten int
}

// Result is the outcome of Learn for one sample.
type Result struct {
	// Output is the network output before training on the sample.
	Output []float64
	// Label is the predicted class (classification only).
	Label string
	// Error is 0 or 1 for classification and the absolute error for regression.
	Error float64
	// Drift reports whether this sample completed the detection of a drift.
	Drift bool
}

// remember stores a sample, bounded by the configured memory.
func (o *Online) remember(sample *learn.Sample) error {
	o.Seen++
	if o.Config.Memory == Reservoir && o.Data.Len() >= o.Config.MemorySize {
		// Algorithm R: keep the new sample with probability size/seen.
		if j := o.random().IntN(o.Seen); j < o.Config.MemorySize {
			if err := o.Data.Replace(j, sample); err != nil {
				return fmt.Errorf("online: %w", err)
			}
		}
		return nil
	}
	if err := o.Data.Add(sample); err != nil {
		return fmt.Errorf("online: %w", err)
	}
	if o.Config.Memory == Window && o.Data.Len() > o.Config.MemorySize {
		o.Data.DropOldest(o.Data.Len() - o.Config.MemorySize)
	}
	return nil
}

// detect feeds the error to the drift detector and reports drift. Only a
// rising error counts: a falling error means the model got better.
func (o *Online) detect(e float64) bool {
	switch o.Config.Drift {
	case ADWIN:
		if o.DriftADWIN == nil {
			o.DriftADWIN = drift.NewADWIN(o.Config.DriftDelta)
		}
		return o.DriftADWIN.Add(e) && o.DriftADWIN.Change > 0
	case PageHinkley:
		if o.DriftPageHinkley == nil {
			o.DriftPageHinkley = &drift.PageHinkley{Threshold: o.Config.DriftDelta}
		}
		return o.DriftPageHinkley.Add(e)
	}
	return false
}

// forget reacts to a drift: it keeps only the samples of the new concept.
// With ADWIN these are as many recent samples as its window holds after the
// change, with Page-Hinkley MinimumDataPoints. A reservoir has no time order
// and is cleared.
func (o *Online) forget() DriftEvent {
	keep := o.Config.MinimumDataPoints
	if o.DriftADWIN != nil && o.Config.Drift == ADWIN {
		keep = o.DriftADWIN.Width
	}
	if o.Config.Memory == Reservoir {
		keep = 0
		o.Seen = 0
	}
	keep = min(keep, o.Data.Len())
	drop := o.Data.Len() - keep
	o.Data.DropOldest(drop)
	return DriftEvent{Kept: keep, Forgotten: drop}
}

// Learn processes one sample of a stream with the test-then-train protocol:
// it predicts the sample, records the result in Prequential and the drift
// detector, reacts to a drift by forgetting old samples, and then stores the
// sample and trains the network on it with Config.HotShots steps.
// Duplicates are stored like any other sample.
func (o *Online) Learn(sample *learn.Sample) (Result, error) {
	if sample == nil {
		return Result{}, fmt.Errorf("online: nil sample")
	}
	if len(sample.Vector) != o.Network.InputSize() {
		return Result{}, fmt.Errorf("online: %w: sample has %d features, network expects %d", nn.ErrDimension, len(sample.Vector), o.Network.InputSize())
	}
	if err := o.Data.Prepare(sample); err != nil {
		return Result{}, fmt.Errorf("online: %w", err)
	}
	out, err := o.Network.Predict(sample.Vector)
	if err != nil {
		return Result{}, fmt.Errorf("online: %w", err)
	}
	if o.Prequential == nil {
		window := o.Config.PrequentialWindow
		if window < 1 {
			window = 500
		}
		if o.Prequential, err = evaluation.NewPrequential(o.Data.Task, o.Data.Labels, window); err != nil {
			return Result{}, err
		}
	}
	o.Learned++
	r := Result{Output: out}
	if o.Data.Task == nn.Classification {
		p := nn.ArgMax(out)
		r.Label = o.Data.Labels[p]
		if r.Error, err = o.Prequential.AddClassification(out, p, sample.Class); err != nil {
			return Result{}, fmt.Errorf("online: %w", err)
		}
	} else {
		r.Error = o.Prequential.AddRegression(sample.Value, out[0])
	}

	// ADWIN may cut its window several times while the new concept
	// establishes itself; the memory follows every cut, but one change is
	// reported only once per DriftCooldown samples.
	if o.detect(r.Error) {
		ev := o.forget()
		if o.Drifts == 0 || o.Learned-o.LastDrift > o.Config.DriftCooldown {
			r.Drift = true
			o.Drifts++
			o.LastDrift = o.Learned
			ev.Sample = o.Learned
			o.logger().Info("drift detected", "sample", ev.Sample, "kept", ev.Kept, "forgotten", ev.Forgotten)
			if o.Config.OnDrift != nil {
				o.Config.OnDrift(ev)
			}
		}
	}
	if err := o.remember(sample); err != nil {
		return Result{}, err
	}
	return r, o.hotShots(sample)
}
