// Package synthetic reads the synthetic data sets of gopher-learn-data.
//
// The data sets are grouped by purpose in folders (classification,
// regression, online, text, loading, benchmark). Datasets lists every file
// with its metadata; Read streams the rows of a file.
//
// The package only uses the standard library. To train models on the data,
// load the files with github.com/breskos/gopher-learn (see README.md).
package synthetic

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// Format is the file format of a data set.
type Format string

const (
	// CSV files have a header row; see Dataset.Comma for the delimiter.
	CSV Format = "csv"
	// Phrase files hold one "text#LABEL" per line.
	Phrase Format = "phrase"
	// LIBSVM files hold sparse rows: "<label> <index>:<value> ...", indices start at 1.
	LIBSVM Format = "libsvm"
)

// Dataset describes one data file.
type Dataset struct {
	// Path is the file path relative to this folder, e.g. "online/abrupt-drift.csv".
	Path string
	// Folder is the purpose of the data set: classification, regression, online, service, text, loading or benchmark.
	Folder      string
	Description string
	// Task is "classification" or "regression".
	Task   string
	Format Format
	// Rows is the number of rows (without the header).
	Rows int
	// Header, Target, Ignore and Comma describe CSV files.
	Header []string
	Target string
	Ignore []string
	Comma  rune
	// Ordered marks streams that must be read in order.
	Ordered bool
	// Applied marks data that models real sensors.
	Applied bool
	// AllowMissing marks files with empty fields on purpose.
	AllowMissing bool
	// RawTarget marks regression files whose target is in raw units
	// instead of [0, 1].
	RawTarget bool
}

// Lookup returns the data set with the given path.
func Lookup(path string) (Dataset, bool) {
	for _, d := range Datasets {
		if d.Path == path {
			return d, true
		}
	}
	return Dataset{}, false
}

// InFolder returns the data sets of one folder, e.g. "online".
func InFolder(folder string) []Dataset {
	var out []Dataset
	for _, d := range Datasets {
		if d.Folder == folder {
			out = append(out, d)
		}
	}
	return out
}

// Features returns the names of the feature columns of a CSV data set:
// all columns except the target and the ignored ones.
func (d Dataset) Features() []string {
	var out []string
	for _, h := range d.Header {
		if h != d.Target && !slices.Contains(d.Ignore, h) {
			out = append(out, h)
		}
	}
	return out
}

// Record is one row of a data set.
type Record struct {
	// Features holds the feature values of CSV rows in the order of
	// Dataset.Features. Empty fields are NaN.
	Features []float64
	// Sparse holds the values of LIBSVM rows by 1-based index.
	Sparse map[int]float64
	// Text holds the text of phrase rows.
	Text string
	// Target is the class label or the regression target as written in the file.
	Target string
	// Ignored holds the values of ignored CSV columns, e.g. t or concept.
	Ignored map[string]string
}

// Read reads the file of the data set below dir (the folder containing
// classification, regression, ...) and calls fn for every row, in file
// order. It stops at the first error returned by fn.
func (d Dataset) Read(dir string, fn func(Record) error) error {
	f, err := os.Open(filepath.Join(dir, filepath.FromSlash(d.Path)))
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	switch d.Format {
	case CSV:
		return d.readCSV(f, fn)
	case Phrase:
		return readLines(f, func(line string) error {
			text, label, ok := strings.Cut(line, "#")
			if !ok {
				return fmt.Errorf("missing '#' in %q", line)
			}
			return fn(Record{Text: text, Target: label})
		})
	case LIBSVM:
		return readLines(f, func(line string) error {
			fields := strings.Fields(line)
			if len(fields) == 0 {
				return errors.New("empty line")
			}
			r := Record{Target: fields[0], Sparse: make(map[int]float64, len(fields)-1)}
			for _, p := range fields[1:] {
				idx, val, ok := strings.Cut(p, ":")
				if !ok {
					return fmt.Errorf("invalid feature %q", p)
				}
				i, err := strconv.Atoi(idx)
				if err != nil {
					return err
				}
				v, err := strconv.ParseFloat(val, 64)
				if err != nil {
					return err
				}
				r.Sparse[i] = v
			}
			return fn(r)
		})
	}
	return fmt.Errorf("unknown format %q", d.Format)
}

func (d Dataset) readCSV(r io.Reader, fn func(Record) error) error {
	cr := csv.NewReader(bufio.NewReader(r))
	if d.Comma != 0 {
		cr.Comma = d.Comma
	}
	header, err := cr.Read()
	if err != nil {
		return fmt.Errorf("%s: header: %w", d.Path, err)
	}
	if !slices.Equal(header, d.Header) {
		return fmt.Errorf("%s: header %v, want %v", d.Path, header, d.Header)
	}
	features := d.Features()
	for line := 2; ; line++ {
		rec, err := cr.Read()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return fmt.Errorf("%s: %w", d.Path, err)
		}
		r := Record{Features: make([]float64, 0, len(features)), Ignored: map[string]string{}}
		for i, name := range header {
			switch {
			case name == d.Target:
				r.Target = rec[i]
			case slices.Contains(d.Ignore, name):
				r.Ignored[name] = rec[i]
			case rec[i] == "":
				r.Features = append(r.Features, math.NaN())
			default:
				v, err := strconv.ParseFloat(rec[i], 64)
				if err != nil {
					return fmt.Errorf("%s: line %d column %q: %w", d.Path, line, name, err)
				}
				r.Features = append(r.Features, v)
			}
		}
		if err := fn(r); err != nil {
			return err
		}
	}
}

func readLines(r io.Reader, fn func(string) error) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		if err := fn(sc.Text()); err != nil {
			return err
		}
	}
	return sc.Err()
}
