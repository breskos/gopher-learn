package main

import (
	"os"
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
	for _, want := range []string{"-seed 3", "-scale 0.001", "## stream/drift.csv", "Delimiter: `;`", "Ignore: `id`"} {
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
