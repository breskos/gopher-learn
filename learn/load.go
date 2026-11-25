package learn

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/breskos/gopher-learn/nn"
)

// LoadCSV reads samples from a CSV file. See ReadCSV.
func (s *Set) LoadCSV(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	return s.ReadCSV(f)
}

// ReadCSV reads samples in CSV format. All columns except the last are
// features. The last column is the class label for classification or the
// target value for regression. Unknown classes are registered in order of
// appearance.
func (s *Set) ReadCSV(r io.Reader) error {
	cr := csv.NewReader(bufio.NewReader(r))
	cr.ReuseRecord = true
	for line := 1; ; line++ {
		record, err := cr.Read()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if len(record) < 2 {
			return fmt.Errorf("learn: line %d: need at least one feature and a target", line)
		}
		last := len(record) - 1
		vector := make([]float64, last)
		for i, field := range record[:last] {
			if vector[i], err = strconv.ParseFloat(strings.TrimSpace(field), 64); err != nil {
				return fmt.Errorf("learn: line %d column %d: %w", line, i+1, err)
			}
		}
		if err := s.addParsed(vector, strings.TrimSpace(record[last])); err != nil {
			return fmt.Errorf("learn: line %d: %w", line, err)
		}
	}
}

// LoadSVM reads samples from a file in LIBSVM format. See ReadSVM.
func (s *Set) LoadSVM(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	return s.ReadSVM(f)
}

// Limits for ReadSVM. Samples are stored as dense vectors, so a single large
// feature index would otherwise allocate a huge vector for every row.
// Variables instead of constants so tests can lower them.
var (
	// maxSVMFeatures is the highest accepted feature index.
	maxSVMFeatures = 1 << 20
	// maxDenseValues limits rows × features of a LIBSVM input (1 GiB of float64).
	maxDenseValues = 1 << 27
)

// ReadSVM reads samples in sparse LIBSVM format ("<target> <index>:<value> ...").
// Indices are 1-based; missing features are zero. The vector length is the
// highest index found in the input (or the input size of a non-empty set).
//
// Because samples are stored densely, feature indices above 1,048,576 and
// inputs with more than 134,217,728 values (rows × features) are rejected.
func (s *Set) ReadSVM(r io.Reader) error {
	type row struct {
		target   string
		features map[int]float64
	}
	var rows []row
	highest := s.InputSize()
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for line := 1; sc.Scan(); line++ {
		fields := strings.Fields(sc.Text())
		if len(fields) == 0 || strings.HasPrefix(fields[0], "#") {
			continue
		}
		rw := row{target: fields[0], features: make(map[int]float64, len(fields)-1)}
		for _, f := range fields[1:] {
			idxStr, valStr, ok := strings.Cut(f, ":")
			if !ok {
				return fmt.Errorf("learn: line %d: invalid feature %q", line, f)
			}
			idx, err := strconv.Atoi(idxStr)
			if err != nil || idx < 1 {
				return fmt.Errorf("learn: line %d: invalid index %q", line, idxStr)
			}
			if idx > maxSVMFeatures {
				return fmt.Errorf("learn: line %d: feature index %d exceeds the limit of %d", line, idx, maxSVMFeatures)
			}
			val, err := strconv.ParseFloat(valStr, 64)
			if err != nil {
				return fmt.Errorf("learn: line %d: %w", line, err)
			}
			rw.features[idx] = val
			highest = max(highest, idx)
		}
		rows = append(rows, rw)
	}
	if err := sc.Err(); err != nil {
		return err
	}
	if len(rows) > 0 && highest > maxDenseValues/len(rows) {
		return fmt.Errorf("learn: %d rows with %d features exceed the limit of %d values", len(rows), highest, maxDenseValues)
	}
	for i, rw := range rows {
		vector := make([]float64, highest)
		for idx, v := range rw.features {
			vector[idx-1] = v
		}
		if err := s.addParsed(vector, rw.target); err != nil {
			return fmt.Errorf("learn: row %d: %w", i+1, err)
		}
	}
	return nil
}

func (s *Set) addParsed(vector []float64, target string) error {
	switch s.Task {
	case nn.Classification:
		if _, ok := s.ClassIndex(target); !ok {
			if err := s.AddClass(target); err != nil {
				return err
			}
		}
		return s.Add(NewClassificationSample(vector, target))
	case nn.Regression:
		v, err := strconv.ParseFloat(target, 64)
		if err != nil {
			return fmt.Errorf("invalid regression target: %w", err)
		}
		return s.Add(NewRegressionSample(vector, v))
	default:
		return fmt.Errorf("invalid task %v", s.Task)
	}
}
