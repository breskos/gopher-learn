package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/breskos/gopher-learn/internal/synth"
)

func TestRun(t *testing.T) {
	dir := t.TempDir()
	if err := run(dir, 3, 0.001, ""); err != nil {
		t.Fatal(err)
	}
	for _, d := range synth.All() {
		info, err := os.Stat(filepath.Join(dir, filepath.FromSlash(d.Path)))
		if err != nil {
			t.Fatalf("%s: %v", d.Path, err)
		}
		if info.Size() == 0 {
			t.Fatalf("%s is empty", d.Path)
		}
	}
	readme, err := os.ReadFile(filepath.Join(dir, "README.md"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"-seed 3", "-scale 0.001", "## online/abrupt-drift.csv", "Delimiter: `;`", "Ignore: `id`"} {
		if !strings.Contains(string(readme), want) {
			t.Errorf("README misses %q", want)
		}
	}
	// no temporary files are left behind
	_ = filepath.WalkDir(dir, func(path string, _ os.DirEntry, _ error) error {
		if strings.Contains(filepath.Base(path), ".tmp") {
			t.Errorf("temporary file left: %s", path)
		}
		return nil
	})
}

func TestRunOnlyAndErrors(t *testing.T) {
	dir := t.TempDir()
	if err := run(dir, 1, 0.01, "spirals"); err != nil {
		t.Fatal(err)
	}
	entries, _ := os.ReadDir(filepath.Join(dir, "classification"))
	if len(entries) != 1 || entries[0].Name() != "spirals.csv" {
		t.Fatalf("only spirals expected, got %v", entries)
	}
	if err := run(dir, 1, 1, "does-not-exist"); err == nil {
		t.Fatal("expected error for an unknown filter")
	}
	// a file where a directory is needed makes writing fail
	blocker := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(blocker, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := run(blocker, 1, 0.01, "blobs"); err == nil {
		t.Fatal("expected error when the output path is a file")
	}
}

// TestGeneratedModule runs vet and the tests of the generated Go module, so
// a change to a generator cannot break the reader, catalog or examples.
func TestGeneratedModule(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the go tool")
	}
	goTool, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go tool not found")
	}
	dir := t.TempDir()
	if err := run(dir, 1, 0.02, ""); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"go.mod", "catalog.go", "reader.go", "synthetic_test.go", "example_test.go"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
	}
	for _, args := range [][]string{{"vet", "./..."}, {"test", "-count=1", "./..."}} {
		cmd := exec.Command(goTool, args...)
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GOFLAGS=-mod=mod", "GOWORK=off")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("go %s failed: %v\n%s", strings.Join(args, " "), err, out)
		}
	}
}

func TestOnlySkipsModule(t *testing.T) {
	dir := t.TempDir()
	if err := run(dir, 1, 0.01, "blobs"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "catalog.go")); !os.IsNotExist(err) {
		t.Fatal("a partial run must not write the catalog")
	}
}
