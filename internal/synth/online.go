package synth

import (
	"math"
	"math/rand/v2"
	"strconv"
	"time"

	"github.com/breskos/gopher-learn/nn"
)

// Streams for online learning. Rows are in time order and carry the concept
// that generated them, so drift detection can be evaluated.

var streamHeader = []string{"t", "x1", "x2", "x3", "x4", "concept", "label"}

// conceptLabel returns the label of a point under concept 1 (x1 + x2 > 0) or
// concept 2 (x1 - x2 > 0), with 5 % label noise.
func conceptLabel(rng *rand.Rand, x [4]float64, concept int) string {
	a := x[0]+x[1] > 0
	if concept == 2 {
		a = x[0]-x[1] > 0
	}
	if rng.Float64() < 0.05 {
		a = !a
	}
	if a {
		return "A"
	}
	return "B"
}

func streamRow(rng *rand.Rand, row []string, t, concept int) {
	var x [4]float64
	for i := range x {
		x[i] = rng.Float64()*2 - 1
	}
	row[0] = strconv.Itoa(t)
	for i, v := range x {
		row[i+1] = num(v)
	}
	row[5] = strconv.Itoa(concept)
	row[6] = conceptLabel(rng, x, concept)
}

func gradualDrift() Dataset {
	return Dataset{
		Path: "online/gradual-drift.csv",
		Description: "Like abrupt-drift, but the change is gradual: between 40 % and 60 % of the stream the probability that a row " +
			"follows concept 2 (label A if x1 - x2 > 0) instead of concept 1 (A if x1 + x2 > 0) rises linearly from 0 to 1. " +
			"x3 and x4 are noise, 5 % of the labels are flipped. Ignore t and concept for training.",
		Task:     nn.Classification,
		Rows:     100_000,
		Header:   streamHeader,
		Target:   "label",
		Ignore:   []string{"t", "concept"},
		Loadable: true,
		Ordered:  true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			row := make([]string, len(streamHeader))
			for t := range n {
				p := clamp01((float64(t)/float64(n) - 0.4) / 0.2)
				concept := 1
				if rng.Float64() < p {
					concept = 2
				}
				streamRow(rng, row, t, concept)
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func recurringConcepts() Dataset {
	return Dataset{
		Path: "online/recurring-concepts.csv",
		Description: "Two concepts that take turns, like seasons: the stream switches between concept 1 (A if x1 + x2 > 0) and " +
			"concept 2 (A if x1 - x2 > 0) every eighth of the stream. A learner that remembers old concepts recovers faster when they return. " +
			"x3 and x4 are noise, 5 % of the labels are flipped. Ignore t and concept for training.",
		Task:     nn.Classification,
		Rows:     100_000,
		Header:   streamHeader,
		Target:   "label",
		Ignore:   []string{"t", "concept"},
		Loadable: true,
		Ordered:  true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			row := make([]string, len(streamHeader))
			period := max(n/8, 1)
			for t := range n {
				concept := 1 + (t/period)%2
				streamRow(rng, row, t, concept)
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func incrementalDrift() Dataset {
	const features = 5
	header := append(append([]string{"t"}, featureNames(features)...), "y")
	return Dataset{
		Path: "online/incremental-drift.csv",
		Description: "Regression with a target that changes slowly over the whole stream: y = sigmoid(w(t) · x) with x1–x5 uniform in [-1, 1] " +
			"and weights that move linearly from w = (2, -1, 1, 0, 0) to w = (-1, 1, 0, 2, -2). " +
			"A model trained only on the beginning gets steadily worse; an online learner has to keep up. Small noise is added. Ignore t.",
		Task:     nn.Regression,
		Rows:     50_000,
		Header:   header,
		Target:   "y",
		Ignore:   []string{"t"},
		Loadable: true,
		Ordered:  true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			from := [features]float64{2, -1, 1, 0, 0}
			to := [features]float64{-1, 1, 0, 2, -2}
			row := make([]string, len(header))
			for t := range n {
				a := float64(t) / float64(n)
				var s float64
				row[0] = strconv.Itoa(t)
				for i := range features {
					x := rng.Float64()*2 - 1
					s += ((1-a)*from[i] + a*to[i]) * x
					row[1+i] = num(x)
				}
				row[1+features] = num(clamp01(sigmoid(s) + rng.NormFloat64()*0.02))
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}

func airQualityAging() Dataset {
	header := []string{"timestamp", "sensor_unit", "hour", "raw_pm25", "temperature_c", "humidity_pct", "pm25_reference"}
	return Dataset{
		Path: "online/air-quality-aging.csv",
		Description: "A low-cost particle sensor (e.g. SDS011) running for two years next to a reference instrument, one row per hour. " +
			"Dust in the optics makes the sensor read higher over time (its gain grows from 1.2 to 1.7), and after 16 months it is replaced " +
			"by a new unit with a different gain and offset (sensor_unit changes from 1 to 2). Humidity still causes overestimation. " +
			"An online calibration model must follow the slow drift and recover after the replacement. " +
			"The target pm25_reference is the reference value divided by 150 µg/m³ and clipped to [0, 1]. Ignore timestamp and sensor_unit for training.",
		Task:     nn.Regression,
		Rows:     2 * 365 * 24,
		Header:   header,
		Target:   "pm25_reference",
		Ignore:   []string{"timestamp", "sensor_unit"},
		Loadable: true,
		Ordered:  true,
		Applied:  true,
		generate: func(rng *rand.Rand, n int, emit func([]string) error) error {
			replaced := n * 2 / 3
			row := make([]string, len(header))
			for i := range n {
				t := epoch.Add(time.Duration(i) * time.Hour)
				hour := float64(t.Hour())
				day := float64(t.YearDay())
				winter := 0.5 + 0.5*math.Cos((day-15)/365*2*math.Pi)
				traffic := math.Exp(-sq(hour-8)/4) + 0.8*math.Exp(-sq(hour-18)/5)
				truePM := (6 + 14*winter + 8*traffic) * logNormal(rng, 0.3)
				temp := 10 - 9*math.Cos((day-15)/365*2*math.Pi) + 4*math.Sin((hour-9)/24*2*math.Pi) + rng.NormFloat64()*1.5
				rh := clampRange(70-1.2*(temp-10)+rng.NormFloat64()*12, 20, 99)
				f := math.Min(rh, 95) / 100
				growth := 1 + 0.25*f*f/(1-f)
				unit, gain, offset := 1, 1.2+0.5*float64(i)/float64(replaced), 1.5
				if i >= replaced {
					unit, gain, offset = 2, 0.9+0.2*float64(i-replaced)/float64(n-replaced), -0.5
				}
				raw := truePM*gain*growth + offset + rng.NormFloat64()*1.5
				row[0] = stamp(t)
				row[1] = strconv.Itoa(unit)
				row[2] = num(hour)
				row[3] = num(math.Max(0, raw))
				row[4] = num(temp)
				row[5] = num(rh)
				row[6] = num(clamp01(truePM / 150))
				if err := emit(row); err != nil {
					return err
				}
			}
			return nil
		},
	}
}
