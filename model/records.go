package model

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

// ReadOption configures ReadRecords.
type ReadOption func(*csv.Reader)

// WithComma sets the field delimiter, e.g. ';'. The default is ','.
func WithComma(r rune) ReadOption {
	return func(c *csv.Reader) { c.Comma = r }
}

// ReadRecords reads CSV data whose first row names the columns and returns
// one record per following row. Every row must have as many fields as the
// header.
func ReadRecords(r io.Reader, opts ...ReadOption) ([]Record, error) {
	cr := csv.NewReader(r)
	cr.ReuseRecord = true
	for _, o := range opts {
		o(cr)
	}
	header, err := cr.Read()
	if errors.Is(err, io.EOF) {
		return nil, errors.New("model: no header row")
	}
	if err != nil {
		return nil, fmt.Errorf("model: %w", err)
	}
	names := make([]string, len(header))
	seen := make(map[string]bool, len(header))
	for i, h := range header {
		n := strings.TrimSpace(h)
		if n == "" {
			return nil, fmt.Errorf("model: column %d has no name", i)
		}
		if seen[n] {
			return nil, fmt.Errorf("model: duplicate column %q", n)
		}
		seen[n] = true
		names[i] = n
	}
	var records []Record
	for {
		row, err := cr.Read()
		if errors.Is(err, io.EOF) {
			return records, nil
		}
		if err != nil {
			return nil, fmt.Errorf("model: %w", err)
		}
		rec := make(Record, len(names))
		for i, v := range row {
			rec[names[i]] = v
		}
		records = append(records, rec)
	}
}

// LoadRecords reads records from a CSV file; see ReadRecords.
func LoadRecords(path string, opts ...ReadOption) ([]Record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	return ReadRecords(f, opts...)
}
