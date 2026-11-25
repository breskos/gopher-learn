package learn

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/breskos/gopher-learn/nn"
)

func writeFile(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadFiles(t *testing.T) {
	s := NewSet(nn.Classification)
	if err := s.LoadCSV(writeFile(t, "data.csv", "1,2,A\n3,4,B\n")); err != nil {
		t.Fatal(err)
	}
	if s.Len() != 2 {
		t.Fatalf("csv: %d samples", s.Len())
	}
	s = NewSet(nn.Regression)
	if err := s.LoadSVM(writeFile(t, "data.svm", "0.5 1:1\n0.7 2:1\n")); err != nil {
		t.Fatal(err)
	}
	if s.Len() != 2 || s.Samples[1].Value != 0.7 {
		t.Fatalf("svm: %+v", s.Samples)
	}
	missing := filepath.Join(t.TempDir(), "missing")
	if err := NewSet(nn.Classification).LoadCSV(missing); err == nil {
		t.Error("LoadCSV: expected error for missing file")
	}
	if err := NewSet(nn.Classification).LoadSVM(missing); err == nil {
		t.Error("LoadSVM: expected error for missing file")
	}
}

func TestReadSVMErrors(t *testing.T) {
	if err := NewSet(nn.Regression).ReadSVM(stringsReader("abc 1:1\n")); err == nil {
		t.Error("expected error for invalid regression target")
	}
	// missing colon, bad index, index below 1, bad value
	for _, in := range []string{"1 1\n", "1 x:1\n", "1 0:1\n", "1 1:abc\n"} {
		if err := NewSet(nn.Classification).ReadSVM(stringsReader(in)); err == nil {
			t.Errorf("ReadSVM(%q): expected error", in)
		}
	}
}

func TestReadCSVMalformed(t *testing.T) {
	if err := NewSet(nn.Classification).ReadCSV(stringsReader("1\n")); err == nil {
		t.Error("expected error for a single column")
	}
	if err := NewSet(nn.Classification).ReadCSV(stringsReader("1,\"2,A\n")); err == nil {
		t.Error("expected error for broken quoting")
	}
	if err := NewSet(nn.Task(9)).ReadCSV(stringsReader("1,2\n")); err == nil {
		t.Error("expected error for invalid task")
	}
}

func TestDistributionAndOneHot(t *testing.T) {
	s := NewSet(nn.Classification)
	if err := s.ReadCSV(stringsReader("1,A\n2,B\n3,A\n")); err != nil {
		t.Fatal(err)
	}
	d := s.Distribution()
	if d["A"] != 2 || d["B"] != 1 {
		t.Fatalf("distribution = %v", d)
	}
	if _, err := s.OneHot("C"); err == nil {
		t.Error("expected error for unknown class")
	}
	if err := s.AddClass("A"); err == nil {
		t.Error("expected error for duplicate class")
	}
	if err := s.Add(nil); err == nil {
		t.Error("expected error for nil sample")
	}
	if err := NewSet(nn.Task(9)).Add(NewRegressionSample([]float64{1}, 1)); err == nil {
		t.Error("expected error for invalid task")
	}
	if NewSet(nn.Regression).OutputSize() != 1 {
		t.Error("regression sets have one output")
	}
}

func stringsReader(s string) *strings.Reader { return strings.NewReader(s) }
