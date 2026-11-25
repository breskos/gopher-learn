package drift

// PageHinkley is the Page-Hinkley test (Page 1954). It detects an increase
// of the mean, such as a rising error rate, by accumulating how far the
// values lie above their running mean. It is cheap and reacts fast to abrupt
// changes; ADWIN copes better with gradual ones.
//
// After a detection the test starts over. PageHinkley serializes to JSON.
type PageHinkley struct {
	// Delta is the magnitude of changes that are tolerated. Default 0.005.
	Delta float64 `json:"delta"`
	// Threshold (lambda) is the accumulated deviation that triggers drift:
	// higher values mean fewer false alarms and slower detection. Default 50.
	Threshold float64 `json:"threshold"`
	// Alpha is the fading factor of the accumulated deviation, slightly
	// below 1 to forget old values. Default 0.9999.
	Alpha float64 `json:"alpha"`
	// MinSamples is the number of values before drift can be signaled.
	// Default 30.
	MinSamples int `json:"min_samples"`

	Count int     `json:"count"`
	Mean  float64 `json:"mean"`
	Sum   float64 `json:"sum"`
	Min   float64 `json:"min"`
}

// NewPageHinkley returns a Page-Hinkley test with the default parameters.
func NewPageHinkley() *PageHinkley { return &PageHinkley{} }

func (p *PageHinkley) defaults() {
	if p.Delta <= 0 {
		p.Delta = 0.005
	}
	if p.Threshold <= 0 {
		p.Threshold = 50
	}
	if p.Alpha <= 0 || p.Alpha > 1 {
		p.Alpha = 0.9999
	}
	if p.MinSamples <= 0 {
		p.MinSamples = 30
	}
}

// Reset forgets all values but keeps the parameters.
func (p *PageHinkley) Reset() {
	p.Count, p.Mean, p.Sum, p.Min = 0, 0, 0, 0
}

// Add records a value and reports whether drift was detected.
func (p *PageHinkley) Add(x float64) bool {
	p.defaults()
	p.Count++
	p.Mean += (x - p.Mean) / float64(p.Count)
	p.Sum = p.Alpha*p.Sum + x - p.Mean - p.Delta
	p.Min = min(p.Min, p.Sum)
	if p.Count >= p.MinSamples && p.Sum-p.Min > p.Threshold {
		p.Reset()
		return true
	}
	return false
}
