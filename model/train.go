package model

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"

	"github.com/breskos/gopher-learn/encoders"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

// Config configures Train. Only Target is required.
type Config struct {
	// Target is the column to predict.
	Target string
	// Task is Classification (the zero value) or Regression.
	Task nn.Task
	// Ignore lists columns that are not used as input, such as IDs.
	Ignore []string
	// Kinds overrides the detected kind of columns. Without an entry, a
	// column whose values are all numbers is Numeric, otherwise Categorical.
	Kinds map[string]Kind
	// Encodings chooses the encoder of categorical columns. Without an
	// entry, the encoders package picks one from the data.
	Encodings map[string]encoders.Type
	// Encoder configures the encoders. Nil uses encoders.DefaultConfig.
	Encoder *encoders.Config
	// Scale is the method used to scale numeric columns. Default Standardize.
	Scale learn.ScaleMethod
	// NoScaling feeds numeric columns to the network unscaled.
	NoScaling bool
	// Hidden holds the sizes of the hidden layers. Nil means one layer of 32
	// neurons; an empty, non-nil slice means no hidden layer.
	Hidden []int
	// HiddenActivation is used for all hidden layers. Default Sigmoid. The
	// output layer is Softmax for classification and Linear for regression.
	HiddenActivation nn.Activation
	// ValidationRatio is the share of records held out to select the best
	// epoch, stratified by class. Default 0.2; negative disables validation.
	ValidationRatio float64
	// Fit configures training. If Fit.Patience is zero, it is set to 10 when
	// there is validation data; a negative value disables early stopping.
	// Fit.Rand defaults to Rand.
	Fit nn.FitConfig
	// Rand drives the validation split, weight initialization and shuffling.
	// Nil uses a randomly seeded source.
	Rand *rand.Rand
}

// Train builds and trains a model from raw records. It detects the kind of
// every column, fits the encoders and the scaler on the training part only
// and trains a network with nn.Fit. It returns the model and the training
// history.
func Train(ctx context.Context, records []Record, cfg Config) (*Model, *nn.History, error) {
	if len(records) == 0 {
		return nil, nil, errors.New("model: no records")
	}
	if cfg.Target == "" {
		return nil, nil, errors.New("model: no target column")
	}
	if cfg.Task != nn.Classification && cfg.Task != nn.Regression {
		return nil, nil, fmt.Errorf("model: invalid task %v", cfg.Task)
	}
	rng := cfg.Rand
	if rng == nil {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}

	m := &Model{Task: cfg.Task, Target: cfg.Target}
	var err error
	if m.Columns, err = columns(records, cfg); err != nil {
		return nil, nil, err
	}

	// targets
	var labels []string
	values := make([]float64, len(records))
	classes := make([]string, len(records))
	for i, r := range records {
		s, ok := r[cfg.Target]
		s = strings.TrimSpace(s)
		if !ok || s == "" {
			return nil, nil, fmt.Errorf("model: record %d: %w: target %q", i, ErrMissing, cfg.Target)
		}
		if cfg.Task == nn.Regression {
			if values[i], ok = parseNumber(s); !ok {
				return nil, nil, fmt.Errorf("model: record %d: %w: target %q is not a number", i, ErrInvalid, s)
			}
			continue
		}
		classes[i] = s
		if !slices.Contains(labels, s) {
			labels = append(labels, s)
		}
	}
	if cfg.Task == nn.Classification && len(labels) < 2 {
		return nil, nil, fmt.Errorf("model: classification needs at least two classes, got %d", len(labels))
	}
	slices.Sort(labels)

	ratio := cfg.ValidationRatio
	if ratio == 0 {
		ratio = 0.2
	}
	trainIdx, valIdx := split(classes, ratio, rng)
	if len(trainIdx) == 0 {
		return nil, nil, errors.New("model: no records left for training")
	}

	if err := m.fitEncoders(records, trainIdx, cfg); err != nil {
		return nil, nil, err
	}
	if err := m.fitScaler(records, trainIdx, cfg); err != nil {
		return nil, nil, err
	}
	if cfg.Task == nn.Regression {
		ys := make([][]float64, len(trainIdx))
		for i, idx := range trainIdx {
			ys[i] = []float64{values[idx]}
		}
		if m.TargetScaler, err = learn.FitScalerVectors(ys, learn.Standardize); err != nil {
			return nil, nil, err
		}
	}

	width, err := m.width()
	if err != nil {
		return nil, nil, err
	}
	if width == 0 {
		return nil, nil, errors.New("model: the columns encode to no features")
	}
	hidden := cfg.Hidden
	if hidden == nil {
		hidden = []int{32}
	}
	netCfg := nn.Config{Inputs: width, Hidden: hidden, HiddenActivation: cfg.HiddenActivation, Rand: rng}
	if cfg.Task == nn.Classification {
		netCfg.Labels, netCfg.OutputActivation = labels, nn.Softmax
	} else {
		netCfg.Outputs, netCfg.OutputActivation = 1, nn.Linear
	}
	if m.Network, err = nn.New(netCfg); err != nil {
		return nil, nil, err
	}

	data := func(idx []int) (nn.Data, error) {
		var d nn.Data
		for _, i := range idx {
			x, err := m.Encode(records[i])
			if err != nil {
				return d, fmt.Errorf("model: record %d: %w", i, err)
			}
			var y []float64
			if cfg.Task == nn.Classification {
				y = make([]float64, len(labels))
				y[slices.Index(labels, classes[i])] = 1
			} else if y, err = m.TargetScaler.Transform([]float64{values[i]}); err != nil {
				return d, err
			}
			d.X, d.Y = append(d.X, x), append(d.Y, y)
		}
		return d, nil
	}
	train, err := data(trainIdx)
	if err != nil {
		return nil, nil, err
	}
	val, err := data(valIdx)
	if err != nil {
		return nil, nil, err
	}

	fit := cfg.Fit
	if fit.Rand == nil {
		fit.Rand = rng
	}
	switch {
	case fit.Patience < 0:
		fit.Patience = 0
	case fit.Patience == 0 && val.Len() > 0:
		fit.Patience = 10
	}
	h, err := nn.Fit(ctx, m.Network, train, val, fit)
	if err != nil {
		return nil, h, err
	}
	return m, h, nil
}

// columns returns the input columns in sorted order with their kinds.
func columns(records []Record, cfg Config) ([]Column, error) {
	names := map[string]bool{}
	for _, r := range records {
		for k := range r {
			names[k] = true
		}
	}
	if !names[cfg.Target] {
		return nil, fmt.Errorf("model: target column %q not found", cfg.Target)
	}
	for _, n := range cfg.Ignore {
		if !names[n] {
			return nil, fmt.Errorf("model: ignored column %q not found", n)
		}
	}
	for n := range cfg.Kinds {
		if !names[n] {
			return nil, fmt.Errorf("model: column %q in Kinds not found", n)
		}
	}
	var cols []Column
	for n := range names {
		if n == cfg.Target || slices.Contains(cfg.Ignore, n) {
			continue
		}
		kind, ok := cfg.Kinds[n]
		if !ok {
			kind = detect(records, n)
		}
		if kind != Numeric && kind != Categorical {
			return nil, fmt.Errorf("model: column %q has invalid kind %v", n, kind)
		}
		if _, ok := cfg.Encodings[n]; ok && kind != Categorical {
			return nil, fmt.Errorf("model: encoding set for numeric column %q", n)
		}
		cols = append(cols, Column{Name: n, Kind: kind})
	}
	if len(cols) == 0 {
		return nil, errors.New("model: no input columns")
	}
	slices.SortFunc(cols, func(a, b Column) int { return strings.Compare(a.Name, b.Name) })
	return cols, nil
}

// detect returns Numeric if every non-empty value of the column is a number
// and there is at least one, otherwise Categorical.
func detect(records []Record, name string) Kind {
	found := false
	for _, r := range records {
		s := strings.TrimSpace(r[name])
		if s == "" {
			continue
		}
		if _, ok := parseNumber(s); !ok {
			return Categorical
		}
		found = true
	}
	if found {
		return Numeric
	}
	return Categorical
}

// split divides record indices into training and validation, stratified by
// class (all records form one group for regression, where classes are empty).
func split(classes []string, ratio float64, rng *rand.Rand) (train, val []int) {
	groups := map[string][]int{}
	var keys []string
	for i, c := range classes {
		if _, ok := groups[c]; !ok {
			keys = append(keys, c)
		}
		groups[c] = append(groups[c], i)
	}
	slices.Sort(keys)
	for _, k := range keys {
		g := groups[k]
		rng.Shuffle(len(g), func(i, j int) { g[i], g[j] = g[j], g[i] })
		n := 0
		if ratio > 0 {
			n = min(int(float64(len(g))*ratio+0.5), len(g)-1)
		}
		val = append(val, g[:n]...)
		train = append(train, g[n:]...)
	}
	return train, val
}

func (m *Model) fitEncoders(records []Record, idx []int, cfg Config) error {
	for _, c := range m.Columns {
		if c.Kind != Categorical {
			continue
		}
		if m.Encoder == nil {
			m.Encoder = encoders.New("model")
			if cfg.Encoder != nil {
				m.Encoder.Config = *cfg.Encoder
			}
		}
		in := encoders.NewInput(encoders.String)
		for _, i := range idx {
			in.AddString(records[i][c.Name])
		}
		t := cfg.Encodings[c.Name]
		if err := m.Encoder.Scan(c.Name, in, t); err != nil {
			return fmt.Errorf("model: column %q: %w", c.Name, err)
		}
		if err := m.Encoder.Fit(c.Name, in); err != nil {
			return fmt.Errorf("model: column %q: %w", c.Name, err)
		}
	}
	return nil
}

func (m *Model) fitScaler(records []Record, idx []int, cfg Config) error {
	if cfg.NoScaling || m.numericCount() == 0 {
		return nil
	}
	vectors := make([][]float64, len(idx))
	for k, i := range idx {
		v, err := m.numeric(records[i])
		if err != nil {
			return fmt.Errorf("model: record %d: %w", i, err)
		}
		vectors[k] = v
	}
	sc, err := learn.FitScalerVectors(vectors, cfg.Scale)
	if err != nil {
		return err
	}
	m.Scaler = sc
	return nil
}
