package synth

import (
	"bufio"
	"bytes"
	"errors"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

func generate(t *testing.T, d Dataset, n int, seed uint64) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := d.Write(&buf, n, seed); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestDeterministic(t *testing.T) {
	for _, d := range All() {
		t.Run(d.Path, func(t *testing.T) {
			a := generate(t, d, 200, 7)
			b := generate(t, d, 200, 7)
			if !bytes.Equal(a, b) {
				t.Fatal("same seed produced different output")
			}
			if bytes.Equal(a, generate(t, d, 200, 8)) {
				t.Fatal("different seeds produced identical output")
			}
		})
	}
}

func TestRowCountAndMetadata(t *testing.T) {
	paths := map[string]bool{}
	for _, d := range All() {
		if paths[d.Path] {
			t.Fatalf("duplicate path %s", d.Path)
		}
		paths[d.Path] = true
		if d.Description == "" || d.Rows <= 0 {
			t.Errorf("%s: missing description or rows", d.Path)
		}
		lines := bytes.Count(generate(t, d, 123, 1), []byte("\n"))
		want := 123
		if d.Format == CSV {
			want++ // header
		}
		if lines != want {
			t.Errorf("%s: %d lines, want %d", d.Path, lines, want)
		}
	}
}

func TestLoadable(t *testing.T) {
	for _, d := range All() {
		if !d.Loadable {
			if _, err := d.Set(10, 1); err == nil {
				t.Errorf("%s: Set must fail for non-loadable data sets", d.Path)
			}
			continue
		}
		t.Run(d.Path, func(t *testing.T) {
			s, err := d.Set(300, 1)
			if err != nil {
				t.Fatal(err)
			}
			if s.Len() != 300 {
				t.Fatalf("%d samples", s.Len())
			}
			wantFeatures := len(d.Header) - 1 - len(d.Ignore)
			if s.InputSize() != wantFeatures || len(s.Features) != wantFeatures {
				t.Fatalf("input size %d, features %v, want %d", s.InputSize(), s.Features, wantFeatures)
			}
			if d.Task == nn.Regression {
				for _, sm := range s.Samples {
					if sm.Value < 0 || sm.Value > 1 {
						t.Fatalf("regression target %v outside [0, 1]", sm.Value)
					}
				}
			}
		})
	}
}

func TestSparseSVM(t *testing.T) {
	d, err := ByPath("loading/sparse.svm")
	if err != nil {
		t.Fatal(err)
	}
	s := learn.NewSet(nn.Classification)
	if err := s.ReadSVM(bytes.NewReader(generate(t, d, 500, 1))); err != nil {
		t.Fatal(err)
	}
	if s.Len() != 500 || s.InputSize() > 1000 || s.InputSize() < 990 {
		t.Fatalf("%d samples with %d features", s.Len(), s.InputSize())
	}
	dist := s.Distribution()
	if dist["+1"] < 150 || dist["-1"] < 150 {
		t.Fatalf("labels unbalanced: %v", dist)
	}
}

func TestMissingValuesRejected(t *testing.T) {
	d, _ := ByPath("loading/missing-values.csv")
	err := learn.NewSet(nn.Classification).ReadCSV(bytes.NewReader(generate(t, d, 1000, 1)), learn.WithHeader())
	if err == nil || !strings.Contains(err.Error(), "missing value") {
		t.Fatalf("want missing value error, got %v", err)
	}
}

func TestIntents(t *testing.T) {
	d, _ := ByPath("text/intents.phrase")
	sc := bufio.NewScanner(bytes.NewReader(generate(t, d, 400, 1)))
	labels := map[string]int{}
	for sc.Scan() {
		text, label, ok := strings.Cut(sc.Text(), "#")
		if !ok || text == "" || strings.Contains(text, "{") {
			t.Fatalf("bad line %q", sc.Text())
		}
		labels[label]++
	}
	if len(labels) != 4 {
		t.Fatalf("labels = %v", labels)
	}
}

func TestImbalancedRatio(t *testing.T) {
	d, _ := ByPath("classification/imbalanced.csv")
	s, err := d.Set(20_000, 3)
	if err != nil {
		t.Fatal(err)
	}
	ratio := float64(s.Distribution()["pos"]) / float64(s.Len())
	if math.Abs(ratio-0.05) > 0.01 {
		t.Fatalf("positive ratio %.3f, want about 0.05", ratio)
	}
}

func TestDriftChangesConcept(t *testing.T) {
	d, _ := ByPath("online/abrupt-drift.csv")
	s, err := d.Set(4000, 1)
	if err != nil {
		t.Fatal(err)
	}
	// agreement of the label with the first-half rule, per half
	agree := func(samples []*learn.Sample) float64 {
		n := 0
		for _, sm := range samples {
			if (sm.Vector[0]+sm.Vector[1] > 0) == (sm.Label == "A") {
				n++
			}
		}
		return float64(n) / float64(len(samples))
	}
	first, second := agree(s.Samples[:2000]), agree(s.Samples[2000:])
	if first < 0.9 || second > 0.65 {
		t.Fatalf("rule agreement before drift %.2f, after %.2f", first, second)
	}
}

func TestErrors(t *testing.T) {
	if _, err := ByPath("nope.csv"); err == nil {
		t.Error("ByPath: expected error")
	}
	d, _ := ByPath("classification/blobs.csv")
	if err := d.Write(&bytes.Buffer{}, 0, 1); err == nil {
		t.Error("Write: expected error for 0 rows")
	}
	bad := d
	bad.Format = Format(9)
	if err := bad.Write(&bytes.Buffer{}, 1, 1); err == nil {
		t.Error("Write: expected error for invalid format")
	}
	if err := d.Write(failWriter{}, 100_000, 1); err == nil {
		t.Error("Write: expected error from failing writer")
	}
	lines, _ := ByPath("text/intents.phrase")
	if err := lines.Write(failWriter{}, 100_000, 1); err == nil {
		t.Error("Write lines: expected error from failing writer")
	}
	if !slices.Equal(featureNames(2), []string{"x1", "x2"}) {
		t.Error("featureNames")
	}
}

type failWriter struct{}

func (failWriter) Write([]byte) (int, error) { return 0, errors.New("disk full") }
