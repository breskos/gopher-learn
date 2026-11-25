// Command gendata writes the synthetic data sets of internal/synth as files.
//
//	go run ./internal/cmd/gendata -out ../gopher-learn-synthetic-data
//
// Rows are streamed to disk, so memory use stays small for any size.
package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/breskos/gopher-learn/internal/synth"
)

func main() {
	out := flag.String("out", "", "output directory (required)")
	seed := flag.Uint64("seed", 1, "random seed; the same seed produces identical files")
	scale := flag.Float64("scale", 1, "multiplier for the default row counts, e.g. 0.1 for small files")
	only := flag.String("only", "", "generate only data sets whose path contains this text")
	flag.Parse()
	if *out == "" || *scale <= 0 {
		flag.Usage()
		os.Exit(2)
	}
	if err := run(*out, *seed, *scale, *only); err != nil {
		log.Fatal(err)
	}
}

func run(out string, seed uint64, scale float64, only string) error {
	var written []synth.Dataset
	var rows []int
	for _, d := range synth.All() {
		if only != "" && !strings.Contains(d.Path, only) {
			continue
		}
		n := max(1, int(float64(d.Rows)*scale))
		path := filepath.Join(out, filepath.FromSlash(d.Path))
		size, err := writeAtomic(path, func(w io.Writer) error { return d.Write(w, n, seed) })
		if err != nil {
			return fmt.Errorf("%s: %w", d.Path, err)
		}
		fmt.Printf("%-34s %9d rows %8.1f MB\n", d.Path, n, float64(size)/(1<<20))
		written = append(written, d)
		rows = append(rows, n)
	}
	if len(written) == 0 {
		return fmt.Errorf("no data set matches %q", only)
	}
	_, err := writeAtomic(filepath.Join(out, "README.md"), func(w io.Writer) error {
		return writeReadme(w, written, rows, seed, scale)
	})
	return err
}

// writeAtomic writes to a temporary file next to path and renames it, so an
// interrupted run never leaves a half-written file behind.
func writeAtomic(path string, fn func(io.Writer) error) (size int64, err error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return 0, err
	}
	f, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp*")
	if err != nil {
		return 0, err
	}
	defer func() {
		if err != nil {
			_ = f.Close()
			_ = os.Remove(f.Name())
		}
	}()
	if err = fn(f); err != nil {
		return 0, err
	}
	info, err := f.Stat()
	if err != nil {
		return 0, err
	}
	if err = f.Chmod(0o644); err != nil {
		return 0, err
	}
	if err = f.Close(); err != nil {
		return 0, err
	}
	return info.Size(), os.Rename(f.Name(), path)
}

func writeReadme(w io.Writer, sets []synth.Dataset, rows []int, seed uint64, scale float64) error {
	var b strings.Builder
	b.WriteString("# Synthetic data sets for gopher-learn\n\n")
	b.WriteString("All data in this folder is generated; it contains no third-party data. ")
	b.WriteString("Every data set has a known structure, so you can tell how well a model should do.\n\n")
	fmt.Fprintf(&b, "Generated with seed %d and row scale %g:\n\n", seed, scale)
	b.WriteString("```\ngo run ./internal/cmd/gendata -out <dir>")
	if seed != 1 {
		fmt.Fprintf(&b, " -seed %d", seed)
	}
	if scale != 1 {
		fmt.Fprintf(&b, " -scale %g", scale)
	}
	b.WriteString("\n```\n\nrun in the [gopher-learn](https://github.com/breskos/gopher-learn) repository. The same seed produces identical files.\n\n")
	b.WriteString("| File | Task | Rows |\n| --- | --- | --- |\n")
	for i, d := range sets {
		fmt.Fprintf(&b, "| [%s](%s) | %v | %d |\n", d.Path, d.Path, d.Task, rows[i])
	}
	for _, d := range sets {
		fmt.Fprintf(&b, "\n## %s\n\n%s\n", d.Path, d.Description)
		if d.Format == synth.CSV {
			fmt.Fprintf(&b, "\nColumns: `%s`. Target: `%s`.", strings.Join(d.Header, "`, `"), d.Target)
			if len(d.Ignore) > 0 {
				fmt.Fprintf(&b, " Ignore: `%s`.", strings.Join(d.Ignore, "`, `"))
			}
			if d.Comma != 0 {
				fmt.Fprintf(&b, " Delimiter: `%c`.", d.Comma)
			}
			b.WriteString("\n")
		}
	}
	_, err := io.WriteString(w, b.String())
	return err
}
