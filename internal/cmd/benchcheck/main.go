// Command benchcheck compares two outputs of go test -bench and reports
// regressions. It prints a Markdown table of every benchmark present in
// both files (median ns/op of all runs) and exits with status 1 if the
// geometric mean of the ratios exceeds -max-geomean or a single benchmark
// exceeds -max-single.
//
//	go test -run='^$' -bench=. -count=6 ./... > new.txt
//	go run ./internal/cmd/benchcheck -base old.txt -head new.txt
package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
)

func main() {
	base := flag.String("base", "", "benchmark output of the base version")
	head := flag.String("head", "", "benchmark output of the new version")
	maxGeomean := flag.Float64("max-geomean", 0.10, "largest accepted slowdown of the geometric mean (0.10 = 10 %)")
	maxSingle := flag.Float64("max-single", 0.50, "largest accepted slowdown of a single benchmark")
	flag.Parse()
	ok, err := run(*base, *head, *maxGeomean, *maxSingle, os.Stdout)
	if err != nil {
		fmt.Fprintln(os.Stderr, "benchcheck:", err)
		os.Exit(2)
	}
	if !ok {
		os.Exit(1)
	}
}

func run(basePath, headPath string, maxGeomean, maxSingle float64, w io.Writer) (bool, error) {
	base, err := load(basePath)
	if err != nil {
		return false, err
	}
	head, err := load(headPath)
	if err != nil {
		return false, err
	}
	return compare(base, head, maxGeomean, maxSingle, w), nil
}

func load(path string) (map[string]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	return parse(f)
}

// parse returns the median ns/op of every benchmark in go test output.
func parse(r io.Reader) (map[string]float64, error) {
	runs := map[string][]float64{}
	sc := bufio.NewScanner(r)
	pkg := ""
	for sc.Scan() {
		line := sc.Text()
		if p, ok := strings.CutPrefix(line, "pkg: "); ok {
			pkg = p
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 4 || !strings.HasPrefix(fields[0], "Benchmark") {
			continue
		}
		for i := 2; i+1 < len(fields); i++ {
			if fields[i+1] == "ns/op" {
				v, err := strconv.ParseFloat(fields[i], 64)
				if err != nil {
					return nil, fmt.Errorf("invalid line %q", line)
				}
				name := strings.TrimPrefix(fields[0], "Benchmark")
				if k := strings.LastIndex(name, "-"); k > 0 {
					name = name[:k] // drop the GOMAXPROCS suffix
				}
				if pkg != "" {
					name = pkg[strings.LastIndex(pkg, "/")+1:] + "." + name
				}
				runs[name] = append(runs[name], v)
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	med := make(map[string]float64, len(runs))
	for name, v := range runs {
		slices.Sort(v)
		m := v[len(v)/2]
		if len(v)%2 == 0 {
			m = (v[len(v)/2-1] + v[len(v)/2]) / 2
		}
		med[name] = m
	}
	return med, nil
}

// compare writes the comparison and reports whether it is acceptable.
func compare(base, head map[string]float64, maxGeomean, maxSingle float64, w io.Writer) bool {
	var b strings.Builder
	defer func() { _, _ = io.WriteString(w, b.String()) }()
	var names []string
	for name := range head {
		if _, ok := base[name]; ok {
			names = append(names, name)
		}
	}
	slices.Sort(names)
	fmt.Fprintln(&b, "| Benchmark | Base | Head | Change |")
	fmt.Fprintln(&b, "| --- | ---: | ---: | ---: |")
	ok := true
	var logSum float64
	for _, name := range names {
		ratio := head[name] / base[name]
		logSum += math.Log(ratio)
		mark := ""
		if ratio > 1+maxSingle {
			mark, ok = " ⚠️", false
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %+.1f %%%s |\n", name, duration(base[name]), duration(head[name]), (ratio-1)*100, mark)
	}
	if len(names) == 0 {
		fmt.Fprintln(&b, "\nNo common benchmarks.")
		return true
	}
	geo := math.Exp(logSum / float64(len(names)))
	fmt.Fprintf(&b, "\nGeometric mean: %+.1f %% over %d benchmarks.\n", (geo-1)*100, len(names))
	if geo > 1+maxGeomean {
		fmt.Fprintf(&b, "The geometric mean is slower than the accepted %.0f %%.\n", maxGeomean*100)
		ok = false
	}
	return ok
}

func duration(ns float64) string {
	switch {
	case ns >= 1e6:
		return fmt.Sprintf("%.2f ms", ns/1e6)
	case ns >= 1e3:
		return fmt.Sprintf("%.2f µs", ns/1e3)
	}
	return fmt.Sprintf("%.1f ns", ns)
}
