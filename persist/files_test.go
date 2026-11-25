package persist

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/breskos/gopher-learn/encoders"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/online"
)

func TestSetFile(t *testing.T) {
	s := learn.NewSet(nn.Classification)
	if err := s.ReadCSV(strings.NewReader("1,2,A\n3,4,B\n")); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "set.json")
	if err := SaveSet(path, s); err != nil {
		t.Fatal(err)
	}
	got, err := LoadSet(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Len() != 2 || got.Task != nn.Classification || !got.Contains(s.Samples[1]) {
		t.Fatalf("set not restored: %+v", got)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o644 {
		t.Errorf("file mode = %v, want 0644", info.Mode().Perm())
	}
}

func TestOnlineFile(t *testing.T) {
	set := learn.NewSet(nn.Regression)
	o, err := online.New(set, 1, []int{2}, online.DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "online.json")
	if err := SaveOnline(path, o); err != nil {
		t.Fatal(err)
	}
	got, err := LoadOnline(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Network.InputSize() != 1 || got.Data.Task != nn.Regression {
		t.Fatalf("online learner not restored: %+v", got)
	}
}

func TestEncoderFile(t *testing.T) {
	e := encoders.New("enc")
	in := encoders.NewInput(encoders.Floats)
	in.AddFloats([]float64{1, 2})
	in.AddFloats([]float64{2, 1})
	if err := e.Fit("x", in); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "encoder.json")
	if err := SaveEncoder(path, e); err != nil {
		t.Fatal(err)
	}
	got, err := LoadEncoder(path)
	if err != nil {
		t.Fatal(err)
	}
	if v, err := got.EncodeFloats("x", []float64{3, 4}); err != nil || len(v) != 2 {
		t.Fatalf("encoder not restored: %v, %v", v, err)
	}
}

func TestLoadErrors(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "missing.json")
	if _, err := LoadNetwork(missing); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("LoadNetwork: want ErrNotExist, got %v", err)
	}
	if _, err := LoadSet(missing); err == nil {
		t.Error("LoadSet: expected error")
	}
	if _, err := LoadOnline(missing); err == nil {
		t.Error("LoadOnline: expected error")
	}
	if _, err := LoadEncoder(missing); err == nil {
		t.Error("LoadEncoder: expected error")
	}
	if err := SaveSet(filepath.Join(t.TempDir(), "no", "such", "dir", "set.json"), learn.NewSet(nn.Regression)); err == nil {
		t.Error("SaveSet: expected error for missing directory")
	}
}

func TestReadErrors(t *testing.T) {
	tests := map[string]string{
		"not json":      "nope",
		"wrong version": `{"kind":"network","version":99,"data":{}}`,
		"invalid net":   `{"kind":"network","version":1,"data":{"layers":[]}}`,
		"bad payload":   `{"kind":"network","version":1,"data":[1,2]}`,
	}
	for name, doc := range tests {
		if _, err := ReadNetwork(strings.NewReader(doc)); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
	if _, err := ReadNetwork(strings.NewReader(`{"kind":"network","version":99,"data":{}}`)); !errors.Is(err, ErrFormat) {
		t.Errorf("wrong version: want ErrFormat, got %v", err)
	}
	if _, err := ReadOnline(strings.NewReader(`{"kind":"online","version":1,"data":{}}`)); !errors.Is(err, ErrFormat) {
		t.Errorf("empty online learner: want ErrFormat, got %v", err)
	}
	for _, kind := range []string{"set", "online", "encoder"} {
		doc := `{"kind":"network","version":1,"data":{}}`
		var err error
		switch kind {
		case "set":
			_, err = ReadSet(strings.NewReader(doc))
		case "online":
			_, err = ReadOnline(strings.NewReader(doc))
		case "encoder":
			_, err = ReadEncoder(strings.NewReader(doc))
		}
		if !errors.Is(err, ErrFormat) {
			t.Errorf("%s: want ErrFormat for wrong kind, got %v", kind, err)
		}
	}
}

func TestWriteUnencodable(t *testing.T) {
	// Activation(99) cannot be marshaled, so writing must fail and leave no file behind.
	n, _ := nn.New(nn.Config{Inputs: 1, Outputs: 1})
	n.Layers[0].Activation = nn.Activation(99)
	dir := t.TempDir()
	if err := SaveNetwork(filepath.Join(dir, "net.json"), n); err == nil {
		t.Fatal("expected error for unencodable network")
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 0 {
		t.Fatalf("temporary file left behind: %v", entries)
	}
	if err := WriteNetwork(&bytes.Buffer{}, n); err == nil {
		t.Fatal("WriteNetwork: expected error")
	}
}
