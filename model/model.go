// Package model bundles everything needed to go from raw records to a
// prediction: column encoders, a feature scaler and a trained network, saved
// and loaded as one artifact.
//
// Train a model from records (for example read from a CSV file with a header
// row), then call Predict with a raw record in production:
//
//	records, _ := model.LoadRecords("houses.csv")
//	m, _, _ := model.Train(ctx, records, model.Config{Target: "price", Task: nn.Regression})
//	p, _ := m.Predict(model.Record{"rooms": "4", "city": "Berlin"})
//	fmt.Println(p.Value)
//
// Use persist.SaveModel and persist.LoadModel to store the model.
package model

import (
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/breskos/gopher-learn/encoders"
	"github.com/breskos/gopher-learn/evaluation"
	"github.com/breskos/gopher-learn/explain"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

var (
	// ErrMissing is returned when a record lacks the value of a numeric column.
	ErrMissing = errors.New("model: missing value")
	// ErrInvalid is returned when a record holds a value that cannot be used,
	// such as text in a numeric column.
	ErrInvalid = errors.New("model: invalid value")
)

// Record is one raw input row: column name to value. Columns the model does
// not use, including the target, are ignored.
type Record map[string]string

// Kind tells how a column is turned into features.
type Kind int

const (
	// Numeric columns hold numbers. They are scaled and fed to the network.
	Numeric Kind = iota
	// Categorical columns hold text such as categories, tags or short
	// phrases. They are encoded by the encoders package; unknown values are
	// allowed and encode to zeros.
	Categorical
)

var kindNames = [...]string{"numeric", "categorical"}

// String returns the name of the kind.
func (k Kind) String() string {
	if k < 0 || int(k) >= len(kindNames) {
		return fmt.Sprintf("Kind(%d)", int(k))
	}
	return kindNames[k]
}

// MarshalText implements encoding.TextMarshaler.
func (k Kind) MarshalText() ([]byte, error) {
	if k < 0 || int(k) >= len(kindNames) {
		return nil, fmt.Errorf("model: invalid kind %d", int(k))
	}
	return []byte(kindNames[k]), nil
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (k *Kind) UnmarshalText(b []byte) error {
	i := slices.Index(kindNames[:], string(b))
	if i < 0 {
		return fmt.Errorf("model: unknown kind %q", b)
	}
	*k = Kind(i)
	return nil
}

// Column is an input column of a model.
type Column struct {
	Name string `json:"name"`
	Kind Kind   `json:"kind"`
}

// Prediction is the result of Predict.
type Prediction struct {
	// Label is the predicted class (classification only).
	Label string `json:"label,omitempty"`
	// Scores maps every class to its network output (classification only).
	Scores map[string]float64 `json:"scores,omitempty"`
	// Value is the predicted value in the units of the target (regression only).
	Value float64 `json:"value,omitempty"`
}

// Model turns raw records into predictions. Numeric columns come first in the
// feature vector, in column order, followed by the encodings of the
// categorical columns.
type Model struct {
	Task    nn.Task  `json:"task"`
	Target  string   `json:"target"`
	Columns []Column `json:"columns"`
	// Encoder encodes the categorical columns; nil without any.
	Encoder *encoders.Encoder `json:"encoder,omitempty"`
	// Scaler scales the numeric columns; nil without numeric columns or
	// with scaling turned off.
	Scaler *learn.Scaler `json:"scaler,omitempty"`
	// TargetScaler maps regression targets to the range the network
	// learned; nil for classification.
	TargetScaler *learn.Scaler `json:"target_scaler,omitempty"`
	Network      *nn.Network   `json:"network"`
}

// Labels returns the classes of a classification model.
func (m *Model) Labels() []string { return slices.Clone(m.Network.Labels) }

func (m *Model) numericCount() int {
	n := 0
	for _, c := range m.Columns {
		if c.Kind == Numeric {
			n++
		}
	}
	return n
}

// width returns the length of the encoded feature vector.
func (m *Model) width() (int, error) {
	w := m.numericCount()
	for _, c := range m.Columns {
		if c.Kind != Categorical {
			continue
		}
		if m.Encoder == nil {
			return 0, fmt.Errorf("model: categorical column %q without encoder", c.Name)
		}
		d, ok := m.Encoder.Dimensions[c.Name]
		if !ok || d.Model == nil {
			return 0, fmt.Errorf("model: column %q is not fitted", c.Name)
		}
		w += d.Model.Dimensions()
	}
	return w, nil
}

// Validate checks that the parts of the model fit together, e.g. after
// decoding.
func (m *Model) Validate() error {
	if m.Network == nil {
		return errors.New("model: no network")
	}
	if err := m.Network.Validate(); err != nil {
		return err
	}
	if len(m.Columns) == 0 {
		return errors.New("model: no columns")
	}
	seen := make(map[string]bool, len(m.Columns))
	for _, c := range m.Columns {
		if c.Kind != Numeric && c.Kind != Categorical {
			return fmt.Errorf("model: column %q has invalid kind %v", c.Name, c.Kind)
		}
		if seen[c.Name] || c.Name == m.Target {
			return fmt.Errorf("model: duplicate column %q", c.Name)
		}
		seen[c.Name] = true
	}
	w, err := m.width()
	if err != nil {
		return err
	}
	if w != m.Network.InputSize() {
		return fmt.Errorf("%w: model encodes %d features, network expects %d", nn.ErrDimension, w, m.Network.InputSize())
	}
	if m.Scaler != nil {
		if err := m.Scaler.Validate(); err != nil {
			return err
		}
		if m.Scaler.Size() != m.numericCount() {
			return fmt.Errorf("%w: scaler for %d features, model has %d numeric columns", nn.ErrDimension, m.Scaler.Size(), m.numericCount())
		}
	}
	switch m.Task {
	case nn.Classification:
		if len(m.Network.Labels) != m.Network.OutputSize() || m.TargetScaler != nil {
			return errors.New("model: classification network needs one label per output and no target scaler")
		}
	case nn.Regression:
		if m.Network.OutputSize() != 1 {
			return fmt.Errorf("model: regression network has %d outputs, want 1", m.Network.OutputSize())
		}
		if m.TargetScaler != nil {
			if err := m.TargetScaler.Validate(); err != nil {
				return err
			}
			if m.TargetScaler.Size() != 1 {
				return errors.New("model: target scaler must have one feature")
			}
		}
	default:
		return fmt.Errorf("model: invalid task %v", m.Task)
	}
	return nil
}

// parseNumber parses a numeric value; it rejects NaN and infinity.
func parseNumber(s string) (float64, bool) {
	f, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil || math.IsNaN(f) || math.IsInf(f, 0) {
		return 0, false
	}
	return f, true
}

// numeric returns the raw values of the numeric columns.
func (m *Model) numeric(r Record) ([]float64, error) {
	out := make([]float64, 0, len(m.Columns))
	for _, c := range m.Columns {
		if c.Kind != Numeric {
			continue
		}
		s, ok := r[c.Name]
		if !ok || strings.TrimSpace(s) == "" {
			return nil, fmt.Errorf("%w: column %q", ErrMissing, c.Name)
		}
		f, ok := parseNumber(s)
		if !ok {
			return nil, fmt.Errorf("%w: column %q: %q is not a number", ErrInvalid, c.Name, s)
		}
		out = append(out, f)
	}
	return out, nil
}

// Encode turns a raw record into the feature vector the network sees.
func (m *Model) Encode(r Record) ([]float64, error) {
	v, err := m.numeric(r)
	if err != nil {
		return nil, err
	}
	if m.Scaler != nil {
		if v, err = m.Scaler.Transform(v); err != nil {
			return nil, err
		}
	}
	for _, c := range m.Columns {
		if c.Kind != Categorical {
			continue
		}
		if m.Encoder == nil {
			return nil, fmt.Errorf("model: categorical column %q without encoder", c.Name)
		}
		e, err := m.Encoder.EncodeString(c.Name, r[c.Name])
		if err != nil {
			return nil, err
		}
		v = append(v, e...)
	}
	return v, nil
}

// Predict encodes a raw record and runs the network.
func (m *Model) Predict(r Record) (Prediction, error) {
	x, err := m.Encode(r)
	if err != nil {
		return Prediction{}, err
	}
	out, err := m.Network.Predict(x)
	if err != nil {
		return Prediction{}, err
	}
	if m.Task == nn.Regression {
		v := out[0]
		if m.TargetScaler != nil {
			inv, err := m.TargetScaler.Inverse(out)
			if err != nil {
				return Prediction{}, err
			}
			v = inv[0]
		}
		return Prediction{Value: v}, nil
	}
	p := Prediction{Label: m.Network.Labels[nn.ArgMax(out)], Scores: make(map[string]float64, len(out))}
	for i, l := range m.Network.Labels {
		p.Scores[l] = out[i]
	}
	return p, nil
}

// Evaluate predicts every record and compares the result with its target
// value. For regression, threshold is the largest error that counts as
// correct; it is ignored for classification.
func (m *Model) Evaluate(records []Record, threshold float64) (*evaluation.Evaluation, error) {
	e := evaluation.New(m.Task, m.Network.Labels)
	e.Threshold = threshold
	for i, r := range records {
		p, err := m.Predict(r)
		if err != nil {
			return nil, fmt.Errorf("model: record %d: %w", i, err)
		}
		actual, ok := r[m.Target]
		if !ok {
			return nil, fmt.Errorf("model: record %d: %w: target %q", i, ErrMissing, m.Target)
		}
		if m.Task == nn.Regression {
			y, ok := parseNumber(actual)
			if !ok {
				return nil, fmt.Errorf("model: record %d: %w: target %q is not a number", i, ErrInvalid, actual)
			}
			e.AddRegression(y, p.Value)
			continue
		}
		c := slices.Index(m.Network.Labels, strings.TrimSpace(actual))
		if c < 0 {
			return nil, fmt.Errorf("model: record %d: %w: unknown class %q", i, ErrInvalid, actual)
		}
		scores := make([]float64, len(m.Network.Labels))
		for j, l := range m.Network.Labels {
			scores[j] = p.Scores[l]
		}
		e.Add(m.Network.Labels[c], p.Label)
		if err := e.AddScores(scores, c); err != nil {
			return nil, err
		}
	}
	return e, nil
}

// Importance computes the permutation importance of every input column on
// records with targets, most important first: the values of one column are
// shuffled across the records and the loss of quality is measured. Use
// records the model was not trained on. Numeric and categorical columns are
// treated alike, so the importance of a categorical column covers all of
// its encoded features.
func (m *Model) Importance(records []Record, cfg explain.Config) ([]explain.Importance, error) {
	if len(records) < 2 {
		return nil, errors.New("model: need at least two records")
	}
	metric := cfg.Metric
	if metric.Value == nil {
		metric = evaluation.Accuracy
		if m.Task == nn.Regression {
			metric = evaluation.RMSE
		}
	}
	names := make([]string, len(m.Columns))
	for j, c := range m.Columns {
		names[j] = c.Name
	}
	// every worker shuffles the values of its own copy of the records
	newScore := func() explain.ScoreFunc {
		shuffled := make([]Record, len(records))
		for i, r := range records {
			shuffled[i] = maps.Clone(r)
		}
		return func(feature int, order []int) (float64, error) {
			if feature >= 0 {
				name := names[feature]
				for i := range shuffled {
					shuffled[i][name] = records[order[i]][name]
				}
				defer func() {
					for i := range shuffled {
						shuffled[i][name] = records[i][name]
					}
				}()
			}
			e, err := m.Evaluate(shuffled, cfg.RegressionThreshold)
			if err != nil {
				return 0, err
			}
			return metric.Value(e), nil
		}
	}
	return explain.PermuteParallel(names, len(records), newScore, metric, cfg)
}
