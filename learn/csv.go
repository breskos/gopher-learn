package learn

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strconv"
	"strings"
)

// CSVOption configures ReadCSV and LoadCSV.
type CSVOption func(*csvConfig)

type csvConfig struct {
	header     bool
	target     string
	targetCol  int
	hasTarget  bool
	ignore     []string
	ignoreCols []int
	comma      rune
}

// WithHeader treats the first row as column names. The names of the feature
// columns are stored in Set.Features.
func WithHeader() CSVOption {
	return func(c *csvConfig) { c.header = true }
}

// WithTarget selects the target column by name. It implies WithHeader.
func WithTarget(name string) CSVOption {
	return func(c *csvConfig) { c.header, c.target = true, name }
}

// WithTargetColumn selects the target column by its 0-based index. Negative
// values count from the end: -1 is the last column, which is the default.
func WithTargetColumn(index int) CSVOption {
	return func(c *csvConfig) { c.targetCol, c.hasTarget = index, true }
}

// WithIgnore skips the named columns, such as IDs. It implies WithHeader.
func WithIgnore(names ...string) CSVOption {
	return func(c *csvConfig) { c.header, c.ignore = true, append(c.ignore, names...) }
}

// WithIgnoreColumns skips the columns with the given 0-based indices.
// Negative values count from the end.
func WithIgnoreColumns(indices ...int) CSVOption {
	return func(c *csvConfig) { c.ignoreCols = append(c.ignoreCols, indices...) }
}

// WithComma sets the field delimiter, e.g. ';' or '\t'. The default is ','.
func WithComma(r rune) CSVOption {
	return func(c *csvConfig) { c.comma = r }
}

// LoadCSV reads samples from a CSV file. See ReadCSV.
func (s *Set) LoadCSV(path string, opts ...CSVOption) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	return s.ReadCSV(f, opts...)
}

// ReadCSV reads samples in CSV format.
//
// By default there is no header row, the last column is the target and all
// other columns are features. The target is the class label for
// classification or the value for regression. Unknown classes are registered
// in order of appearance. Options select a header, the target column and
// columns to ignore. Empty fields are reported as missing values.
func (s *Set) ReadCSV(r io.Reader, opts ...CSVOption) error {
	cfg := csvConfig{comma: ','}
	for _, o := range opts {
		o(&cfg)
	}
	cr := csv.NewReader(bufio.NewReader(r))
	cr.Comma = cfg.comma
	cr.ReuseRecord = true

	var names []string
	var layout *csvLayout
	for line := 1; ; line++ {
		record, err := cr.Read()
		if errors.Is(err, io.EOF) {
			if layout == nil && cfg.header && names == nil {
				return errors.New("learn: missing header row")
			}
			return nil
		}
		if err != nil {
			return err
		}
		if cfg.header && names == nil {
			names = make([]string, len(record))
			for i, n := range record {
				names[i] = strings.TrimSpace(n)
			}
			if layout, err = cfg.layout(names, len(record)); err != nil {
				return err
			}
			if err := s.setFeatures(layout.featureNames(names)); err != nil {
				return err
			}
			continue
		}
		if layout == nil {
			if layout, err = cfg.layout(nil, len(record)); err != nil {
				return err
			}
		}
		vector := make([]float64, len(layout.features))
		for i, col := range layout.features {
			field := strings.TrimSpace(record[col])
			if field == "" {
				return fmt.Errorf("learn: line %d column %s: missing value", line, columnName(names, col))
			}
			if vector[i], err = strconv.ParseFloat(field, 64); err != nil {
				return fmt.Errorf("learn: line %d column %s: %w", line, columnName(names, col), err)
			}
		}
		target := strings.TrimSpace(record[layout.target])
		if target == "" {
			return fmt.Errorf("learn: line %d column %s: missing target", line, columnName(names, layout.target))
		}
		if err := s.addParsed(vector, target); err != nil {
			return fmt.Errorf("learn: line %d: %w", line, err)
		}
	}
}

type csvLayout struct {
	target   int
	features []int
}

func (l *csvLayout) featureNames(names []string) []string {
	out := make([]string, len(l.features))
	for i, col := range l.features {
		out[i] = names[col]
	}
	return out
}

// layout resolves the target and feature columns for a row of the given width.
func (c *csvConfig) layout(names []string, width int) (*csvLayout, error) {
	resolve := func(i int) (int, error) {
		if i < 0 {
			i += width
		}
		if i < 0 || i >= width {
			return 0, fmt.Errorf("learn: column index out of range for %d columns", width)
		}
		return i, nil
	}
	lookup := func(name string) (int, error) {
		i := slices.Index(names, name)
		if i < 0 {
			return 0, fmt.Errorf("learn: column %q not found in header %v", name, names)
		}
		return i, nil
	}

	l := &csvLayout{target: width - 1}
	var err error
	switch {
	case c.target != "":
		if l.target, err = lookup(c.target); err != nil {
			return nil, err
		}
	case c.hasTarget:
		if l.target, err = resolve(c.targetCol); err != nil {
			return nil, err
		}
	}
	ignored := map[int]bool{}
	for _, name := range c.ignore {
		i, err := lookup(name)
		if err != nil {
			return nil, err
		}
		ignored[i] = true
	}
	for _, idx := range c.ignoreCols {
		i, err := resolve(idx)
		if err != nil {
			return nil, err
		}
		ignored[i] = true
	}
	if ignored[l.target] {
		return nil, fmt.Errorf("learn: target column %s is ignored", columnName(names, l.target))
	}
	for i := range width {
		if i != l.target && !ignored[i] {
			l.features = append(l.features, i)
		}
	}
	if len(l.features) == 0 {
		return nil, errors.New("learn: need at least one feature column besides the target")
	}
	return l, nil
}

// setFeatures records the feature names, or checks them against the names
// of data loaded earlier.
func (s *Set) setFeatures(names []string) error {
	if len(s.Features) > 0 && !slices.Equal(s.Features, names) {
		return fmt.Errorf("learn: feature columns %v do not match %v of the set", names, s.Features)
	}
	if len(s.Features) == 0 && s.Len() > 0 && s.InputSize() != len(names) {
		return fmt.Errorf("learn: %d feature columns, set has %d features", len(names), s.InputSize())
	}
	s.Features = names
	return nil
}

func columnName(names []string, col int) string {
	if col < len(names) && names[col] != "" {
		return strconv.Quote(names[col])
	}
	return strconv.Itoa(col + 1)
}
