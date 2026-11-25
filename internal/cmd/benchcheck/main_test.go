package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const baseOut = `goos: linux
pkg: github.com/breskos/gopher-learn/nn
BenchmarkPredict/tiny-8   	 1000	       100 ns/op	      16 B/op	       1 allocs/op
BenchmarkPredict/tiny-8   	 1000	       120 ns/op	      16 B/op	       1 allocs/op
BenchmarkPredict/tiny-8   	 1000	       110 ns/op	      16 B/op	       1 allocs/op
BenchmarkFit-8   	 10	   2000000 ns/op
BenchmarkOld-8   	 10	   2000 ns/op
PASS
`

func write(t *testing.T, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "bench.txt")
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestParse(t *testing.T) {
	m, err := parse(strings.NewReader(baseOut))
	if err != nil {
		t.Fatal(err)
	}
	if m["nn.Predict/tiny"] != 110 || m["nn.Fit"] != 2e6 || len(m) != 3 {
		t.Fatalf("parsed %v", m)
	}
	m, _ = parse(strings.NewReader("BenchmarkX-8 1 10 ns/op\nBenchmarkX-8 1 20 ns/op\n"))
	if m["X"] != 15 {
		t.Fatalf("median of two: %v", m)
	}
	if _, err := parse(strings.NewReader("BenchmarkX-8 1 abc ns/op\n")); err == nil {
		t.Error("expected error")
	}
}

func TestCompare(t *testing.T) {
	base := write(t, baseOut)
	faster := write(t, strings.NewReplacer("100 ns", "50 ns", "120 ns", "60 ns", "110 ns", "55 ns").Replace(baseOut))
	var out strings.Builder
	ok, err := run(base, faster, 0.1, 0.5, &out)
	if err != nil || !ok {
		t.Fatalf("faster rejected: %v\n%s", err, out.String())
	}
	if !strings.Contains(out.String(), "-50.0 %") || !strings.Contains(out.String(), "110.0 ns") {
		t.Fatalf("table:\n%s", out.String())
	}
	slower := write(t, strings.ReplaceAll(baseOut, "2000000 ns", "4000000 ns"))
	out.Reset()
	if ok, _ := run(base, slower, 0.1, 0.5, &out); ok || !strings.Contains(out.String(), "⚠️") {
		t.Fatalf("regression accepted:\n%s", out.String())
	}
	out.Reset()
	if ok, _ := run(base, slower, 0.1, 2, &out); ok || !strings.Contains(out.String(), "geometric mean is slower") {
		t.Fatalf("geomean regression accepted:\n%s", out.String())
	}
	out.Reset()
	if ok, _ := run(base, write(t, "BenchmarkNew-8 1 5 ns/op\n"), 0.1, 0.5, &out); !ok || !strings.Contains(out.String(), "No common") {
		t.Fatalf("disjoint:\n%s", out.String())
	}
	if _, err := run(filepath.Join(t.TempDir(), "missing"), base, 0.1, 0.5, &out); err == nil {
		t.Error("missing base: expected error")
	}
	if _, err := run(base, filepath.Join(t.TempDir(), "missing"), 0.1, 0.5, &out); err == nil {
		t.Error("missing head: expected error")
	}
	if duration(1.5e6) != "1.50 ms" || duration(1500) != "1.50 µs" {
		t.Error("duration")
	}
}
