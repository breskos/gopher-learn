// Package synth generates synthetic data sets with known properties.
//
// The generators are deterministic for a given seed and stream their rows,
// so large files can be written without holding them in memory. Tests use
// small in-memory versions of the same data sets.
package synth

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"strconv"
	"strings"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

// Format is the file format of a data set.
type Format int

const (
	// CSV writes comma (or Comma) separated records with a header row.
	CSV Format = iota
	// Lines writes the first field of every record as a raw line
	// (used for LIBSVM files and "text#label" phrase files).
	Lines
)

// Dataset describes one synthetic data set.
type Dataset struct {
	// Path is the file path relative to the output directory.
	Path string
	// Description explains what the data set contains and what it tests.
	Description string
	Task        nn.Task
	Format      Format
	// Rows is the default number of rows written to files.
	Rows int
	// Header holds the CSV column names.
	Header []string
	// Target names the target column, Ignore columns that are not features.
	Target string
	Ignore []string
	// Comma is the CSV delimiter; zero means ','.
	Comma rune
	// Loadable reports whether learn.Set.ReadCSV can load the file.
	Loadable bool
	// Ordered marks streams whose rows must be read in order (online learning).
	Ordered bool
	// Applied marks data sets that model real sensors rather than a
	// mathematical benchmark.
	Applied bool
	// AllowMissing marks data sets that contain empty fields on purpose.
	AllowMissing bool

	generate func(rng *rand.Rand, n int, emit func([]string) error) error
}

// Write streams n rows of the data set to w.
func (d Dataset) Write(w io.Writer, n int, seed uint64) error {
	if n <= 0 {
		return fmt.Errorf("synth: %s: row count must be positive, got %d", d.Path, n)
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x5eed))
	bw := bufio.NewWriterSize(w, 1<<16)
	switch d.Format {
	case CSV:
		cw := csv.NewWriter(bw)
		if d.Comma != 0 {
			cw.Comma = d.Comma
		}
		if err := cw.Write(d.Header); err != nil {
			return err
		}
		if err := d.generate(rng, n, cw.Write); err != nil {
			return err
		}
		cw.Flush()
		if err := cw.Error(); err != nil {
			return err
		}
	case Lines:
		err := d.generate(rng, n, func(r []string) error {
			if _, err := bw.WriteString(r[0]); err != nil {
				return err
			}
			return bw.WriteByte('\n')
		})
		if err != nil {
			return err
		}
	default:
		return fmt.Errorf("synth: %s: invalid format %d", d.Path, d.Format)
	}
	return bw.Flush()
}

// Set generates n rows and loads them into a learn.Set, the same way a user
// would load the file. Only loadable data sets can be converted.
func (d Dataset) Set(n int, seed uint64) (*learn.Set, error) {
	if !d.Loadable {
		return nil, fmt.Errorf("synth: %s cannot be loaded as a set", d.Path)
	}
	var buf bytes.Buffer
	if err := d.Write(&buf, n, seed); err != nil {
		return nil, err
	}
	opts := []learn.CSVOption{learn.WithTarget(d.Target), learn.WithIgnore(d.Ignore...)}
	if d.Comma != 0 {
		opts = append(opts, learn.WithComma(d.Comma))
	}
	s := learn.NewSet(d.Task)
	if err := s.ReadCSV(&buf, opts...); err != nil {
		return nil, fmt.Errorf("synth: %s: %w", d.Path, err)
	}
	return s, nil
}

// ByPath returns the data set with the given path.
func ByPath(path string) (Dataset, error) {
	for _, d := range All() {
		if d.Path == path {
			return d, nil
		}
	}
	return Dataset{}, errors.New("synth: unknown data set " + strconv.Quote(path))
}

// num formats a value compactly with six significant digits.
func num(x float64) string { return strconv.FormatFloat(x, 'g', 6, 64) }

// featureNames returns x1..xn.
func featureNames(n int) []string {
	names := make([]string, n)
	for i := range names {
		names[i] = "x" + strconv.Itoa(i+1)
	}
	return names
}

// Folder returns the top-level folder of the data set.
func (d Dataset) Folder() string {
	folder, _, _ := strings.Cut(d.Path, "/")
	return folder
}
