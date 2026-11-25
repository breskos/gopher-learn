package learn

import (
	"errors"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/breskos/gopher-learn/nn"
)

const winesCSV = `id,acidity,sugar,quality
1,7.4,1.9,0.5
2,7.8,2.6,0.6
3,11.2,1.9,0.6
`

func TestReadCSVHeaderAndTarget(t *testing.T) {
	s := NewSet(nn.Regression)
	err := s.ReadCSV(strings.NewReader(winesCSV), WithTarget("quality"), WithIgnore("id"))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(s.Features, []string{"acidity", "sugar"}) {
		t.Fatalf("features = %v", s.Features)
	}
	if s.Len() != 3 || !slices.Equal(s.Samples[2].Vector, []float64{11.2, 1.9}) || s.Samples[2].Value != 0.6 {
		t.Fatalf("samples = %+v", s.Samples[2])
	}
}

func TestReadCSVTargetColumn(t *testing.T) {
	data := "A,1,2\nB,3,4\n"
	s := NewSet(nn.Classification)
	if err := s.ReadCSV(strings.NewReader(data), WithTargetColumn(0)); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(s.Labels, []string{"A", "B"}) || !slices.Equal(s.Samples[1].Vector, []float64{3, 4}) {
		t.Fatalf("labels %v, sample %v", s.Labels, s.Samples[1].Vector)
	}
	if s.Features != nil {
		t.Fatalf("features without header = %v", s.Features)
	}

	s = NewSet(nn.Regression)
	if err := s.ReadCSV(strings.NewReader("9,1,0.5,2\n"), WithTargetColumn(-2), WithIgnoreColumns(0)); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(s.Samples[0].Vector, []float64{1, 2}) || s.Samples[0].Value != 0.5 {
		t.Fatalf("sample = %+v", s.Samples[0])
	}
}

func TestReadCSVComma(t *testing.T) {
	s := NewSet(nn.Classification)
	if err := s.ReadCSV(strings.NewReader("x;y;label\n1,5;2;A\n"), WithHeader(), WithComma(';')); err == nil {
		t.Fatal("expected error: '1,5' is not a number")
	}
	s = NewSet(nn.Classification)
	if err := s.ReadCSV(strings.NewReader("x\ty\tlabel\n1.5\t2\tA\n"), WithHeader(), WithComma('\t')); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(s.Samples[0].Vector, []float64{1.5, 2}) {
		t.Fatalf("vector = %v", s.Samples[0].Vector)
	}
}

func TestReadCSVOptionErrors(t *testing.T) {
	tests := []struct {
		name string
		data string
		opts []CSVOption
	}{
		{"unknown target", winesCSV, []CSVOption{WithTarget("price")}},
		{"unknown ignore", winesCSV, []CSVOption{WithIgnore("color")}},
		{"target ignored", winesCSV, []CSVOption{WithTarget("quality"), WithIgnore("quality")}},
		{"target out of range", "1,2\n", []CSVOption{WithTargetColumn(5)}},
		{"ignore out of range", "1,2\n", []CSVOption{WithIgnoreColumns(-3)}},
		{"no features", "a,b\n1,2\n", []CSVOption{WithHeader(), WithIgnore("a")}},
		{"missing header", "", []CSVOption{WithHeader()}},
		{"missing value", "1,,0.5\n", nil},
		{"missing target", "1,2,\n", nil},
		{"missing value named", "a,b,y\n1,,0.5\n", []CSVOption{WithHeader()}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := NewSet(nn.Regression).ReadCSV(strings.NewReader(tt.data), tt.opts...)
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
	err := NewSet(nn.Regression).ReadCSV(strings.NewReader("a,b,y\n1,,0.5\n"), WithHeader())
	if err == nil || !strings.Contains(err.Error(), `column "b": missing value`) {
		t.Fatalf("error should name the column: %v", err)
	}
	if err := NewSet(nn.Regression).ReadCSV(strings.NewReader("")); err != nil {
		t.Fatalf("empty input without header must be fine: %v", err)
	}
	headerOnly := NewSet(nn.Regression)
	if err := headerOnly.ReadCSV(strings.NewReader("a,b\n"), WithHeader()); err != nil || headerOnly.Len() != 0 {
		t.Fatalf("header without data is an empty set: %d samples, %v", headerOnly.Len(), err)
	}
}

func TestReadCSVSecondFile(t *testing.T) {
	s := NewSet(nn.Regression)
	if err := s.ReadCSV(strings.NewReader("a,b,y\n1,2,0.5\n"), WithHeader()); err != nil {
		t.Fatal(err)
	}
	if err := s.ReadCSV(strings.NewReader("a,b,y\n3,4,0.7\n"), WithHeader()); err != nil {
		t.Fatalf("same columns must be accepted: %v", err)
	}
	if err := s.ReadCSV(strings.NewReader("b,a,y\n3,4,0.7\n"), WithHeader()); err == nil {
		t.Fatal("expected error for different columns")
	}
	unnamed := NewSet(nn.Regression)
	_ = unnamed.ReadCSV(strings.NewReader("1,2,0.5\n"))
	if err := unnamed.ReadCSV(strings.NewReader("a,b,c,y\n1,2,3,0.5\n"), WithHeader()); err == nil {
		t.Fatal("expected error for a different number of features")
	}
	if err := unnamed.ReadCSV(strings.NewReader("a,b,y\n1,2,0.5\n"), WithHeader()); err != nil {
		t.Fatalf("naming the features of an unnamed set must work: %v", err)
	}
	if !slices.Equal(unnamed.Features, []string{"a", "b"}) {
		t.Fatalf("features = %v", unnamed.Features)
	}
}

func TestLoadCSVWithOptions(t *testing.T) {
	path := writeFile(t, "wines.csv", winesCSV)
	s := NewSet(nn.Regression)
	if err := s.LoadCSV(path, WithTarget("quality"), WithIgnore("id")); err != nil {
		t.Fatal(err)
	}
	if s.Len() != 3 {
		t.Fatalf("len = %d", s.Len())
	}
	// Without options the header row is parsed as data and fails.
	if err := NewSet(nn.Regression).LoadCSV(path); !errors.Is(err, strconv.ErrSyntax) {
		t.Fatalf("want a parse error for the header row, got %v", err)
	}
}
