package main

import (
	"bytes"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strconv"

	"github.com/breskos/gopher-learn/internal/synth"
	"github.com/breskos/gopher-learn/nn"
)

// event is one row of an event log (service/offers.csv or
// service/deliveries.csv of gopher-learn-data).
type event struct {
	id       string
	t        int       // seconds
	x        []float64 // raw features
	truth    string    // true label or value
	feedback string    // reported label or value; empty: never
	fbDelay  int       // seconds after t
	corr     int       // seconds after the feedback; -1: no correction
	concept  int
}

// logSpec describes the columns of an event log.
type logSpec struct {
	task     nn.Task
	target   string
	features []string
	labels   []string // classification
}

var specs = map[string]logSpec{
	"offers": {
		task:     nn.Classification,
		target:   "accepted",
		features: []string{"price_ratio", "visits", "basket_eur", "hour", "discount_pct"},
		labels:   []string{"yes", "no"},
	},
	"deliveries": {
		task:     nn.Regression,
		target:   "minutes",
		features: []string{"distance_km", "items", "traffic", "hour", "rain"},
	},
}

// open returns the event log: the file if a path is given, otherwise the
// same data generated in memory.
func open(path, name string, rows int) (io.Reader, error) {
	if path != "" {
		b, err := os.ReadFile(path)
		return bytes.NewReader(b), err
	}
	d, err := synth.ByPath("service/" + name + ".csv")
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	if err := d.Write(&buf, rows, 1); err != nil {
		return nil, err
	}
	return &buf, nil
}

// readEvents parses an event log.
func readEvents(r io.Reader, spec logSpec) ([]event, error) {
	records, err := csv.NewReader(r).ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) < 2 {
		return nil, errors.New("no events")
	}
	col := map[string]int{}
	for i, name := range records[0] {
		col[name] = i
	}
	need := append([]string{"event", "t", "concept", "feedback_delay", "feedback", "correction_delay", spec.target}, spec.features...)
	for _, name := range need {
		if _, ok := col[name]; !ok {
			return nil, fmt.Errorf("column %q missing", name)
		}
	}
	atoi := func(s string, empty int) (int, error) {
		if s == "" {
			return empty, nil
		}
		return strconv.Atoi(s)
	}
	events := make([]event, 0, len(records)-1)
	for line, rec := range records[1:] {
		e := event{id: rec[col["event"]], truth: rec[col[spec.target]], feedback: rec[col["feedback"]]}
		var errs []error
		var err error
		e.t, err = atoi(rec[col["t"]], 0)
		errs = append(errs, err)
		e.concept, err = atoi(rec[col["concept"]], 0)
		errs = append(errs, err)
		e.fbDelay, err = atoi(rec[col["feedback_delay"]], -1)
		errs = append(errs, err)
		e.corr, err = atoi(rec[col["correction_delay"]], -1)
		errs = append(errs, err)
		for _, f := range spec.features {
			v, err := strconv.ParseFloat(rec[col[f]], 64)
			errs = append(errs, err)
			e.x = append(e.x, v)
		}
		if err := errors.Join(errs...); err != nil {
			return nil, fmt.Errorf("line %d: %w", line+2, err)
		}
		if e.fbDelay < 0 {
			e.feedback = ""
		}
		if spec.task == nn.Classification && !slices.Contains(spec.labels, e.truth) {
			return nil, fmt.Errorf("line %d: unknown label %q", line+2, e.truth)
		}
		events = append(events, e)
	}
	return events, nil
}
