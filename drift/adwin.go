package drift

import "math"

// Bucket summarizes Count consecutive values of an ADWIN window.
type Bucket struct {
	Count    int     `json:"count"`
	Total    float64 `json:"total"`
	Variance float64 `json:"variance"` // sum of squared deviations from the bucket mean
}

// ADWIN is the adaptive windowing detector by Bifet and Gavaldà (2007). It
// keeps a window of recent values whose size adapts automatically: when two
// parts of the window have significantly different means, the older part is
// dropped and drift is reported. Its memory grows only logarithmically with
// the window length because values are summarized in exponential buckets.
//
// ADWIN works on any bounded values; the guarantees assume values in [0, 1].
// It serializes to JSON, so a learner can persist its state.
type ADWIN struct {
	// Delta is the confidence: smaller values mean fewer false alarms and
	// slower detection. Default 0.002.
	Delta float64 `json:"delta"`
	// MinWindow is the minimum size of each window part that is compared.
	// Default 5.
	MinWindow int `json:"min_window"`
	// MaxBuckets is the number of buckets per size before two are merged.
	// Default 5.
	MaxBuckets int `json:"max_buckets"`
	// Clock checks for drift every Clock values. Default 1 (every value).
	Clock int `json:"clock"`

	// Levels holds the buckets; level i contains buckets of 2^i values,
	// oldest first.
	Levels   [][]Bucket `json:"levels"`
	Width    int        `json:"width"`
	Total    float64    `json:"total"`
	Variance float64    `json:"variance"` // sum of squared deviations from the mean
	Ticks    int        `json:"ticks"`
	// Change is the difference between the mean of the newer and the older
	// part at the last detected drift: positive if the values rose.
	Change float64 `json:"change"`
}

// NewADWIN returns an ADWIN detector with confidence delta (0 means 0.002).
func NewADWIN(delta float64) *ADWIN {
	if delta <= 0 {
		delta = 0.002
	}
	return &ADWIN{Delta: delta}
}

func (a *ADWIN) defaults() {
	if a.Delta <= 0 {
		a.Delta = 0.002
	}
	if a.MinWindow <= 0 {
		a.MinWindow = 5
	}
	if a.MaxBuckets <= 0 {
		a.MaxBuckets = 5
	}
	if a.Clock <= 0 {
		a.Clock = 1
	}
}

// Mean returns the mean of the current window.
func (a *ADWIN) Mean() float64 {
	if a.Width == 0 {
		return 0
	}
	return a.Total / float64(a.Width)
}

// Reset forgets all values but keeps the parameters.
func (a *ADWIN) Reset() {
	a.Levels, a.Width, a.Total, a.Variance, a.Ticks, a.Change = nil, 0, 0, 0, 0, 0
}

// Add records a value and reports whether drift was detected, in either
// direction; Change tells whether the values rose or fell. After a detection
// the window holds only the values after the change.
func (a *ADWIN) Add(x float64) bool {
	a.defaults()
	if a.Width > 0 {
		d := x - a.Mean()
		a.Variance += float64(a.Width) * d * d / float64(a.Width+1)
	}
	a.Width++
	a.Total += x
	if len(a.Levels) == 0 {
		a.Levels = append(a.Levels, nil)
	}
	a.Levels[0] = append(a.Levels[0], Bucket{Count: 1, Total: x})
	a.compress()

	a.Ticks++
	if a.Ticks%a.Clock != 0 {
		return false
	}
	drift := false
	for a.cut() {
		drift = true
	}
	return drift
}

// compress merges the two oldest buckets of a level into the next level
// while a level holds more than MaxBuckets buckets.
func (a *ADWIN) compress() {
	for i := 0; i < len(a.Levels); i++ {
		if len(a.Levels[i]) <= a.MaxBuckets {
			break
		}
		level := a.Levels[i]
		b1, b2 := level[0], level[1]
		// shift in place to keep the capacity and avoid allocations
		a.Levels[i] = level[:copy(level, level[2:])]
		if i+1 == len(a.Levels) {
			a.Levels = append(a.Levels, nil)
		}
		a.Levels[i+1] = append(a.Levels[i+1], merge(b1, b2))
	}
}

func merge(b1, b2 Bucket) Bucket {
	n1, n2 := float64(b1.Count), float64(b2.Count)
	d := b1.Total/n1 - b2.Total/n2
	return Bucket{
		Count:    b1.Count + b2.Count,
		Total:    b1.Total + b2.Total,
		Variance: b1.Variance + b2.Variance + n1*n2*d*d/(n1+n2),
	}
}

// cut looks for a split of the window into an older and a newer part with
// significantly different means. If it finds one, it drops the oldest bucket
// and returns true.
func (a *ADWIN) cut() bool {
	if a.Width < 2*a.MinWindow {
		return false
	}
	n := float64(a.Width)
	v := a.Variance / n
	dd := math.Log(2 * math.Log(n) / a.Delta)
	var n0, t0 float64
	for i := len(a.Levels) - 1; i >= 0; i-- {
		for _, b := range a.Levels[i] {
			n0 += float64(b.Count)
			t0 += b.Total
			n1 := n - n0
			if n1 < float64(a.MinWindow) {
				return false
			}
			if n0 < float64(a.MinWindow) {
				continue
			}
			diff := t0/n0 - (a.Total-t0)/n1
			m := 1/(n0-float64(a.MinWindow)+1) + 1/(n1-float64(a.MinWindow)+1)
			eps := math.Sqrt(2*m*v*dd) + 2.0/3*dd*m
			if math.Abs(diff) > eps {
				a.Change = -diff
				a.dropOldest()
				return true
			}
		}
	}
	return false
}

func (a *ADWIN) dropOldest() {
	top := len(a.Levels) - 1
	level := a.Levels[top]
	b := level[0]
	a.Levels[top] = level[:copy(level, level[1:])]
	for len(a.Levels) > 0 && len(a.Levels[len(a.Levels)-1]) == 0 {
		a.Levels = a.Levels[:len(a.Levels)-1]
	}
	nb := float64(b.Count)
	ub := b.Total / nb
	a.Width -= b.Count
	a.Total -= b.Total
	if a.Width == 0 {
		a.Total, a.Variance = 0, 0
		return
	}
	n := float64(a.Width)
	d := ub - a.Total/n
	a.Variance = max(a.Variance-b.Variance-nb*n*d*d/(nb+n), 0)
}
