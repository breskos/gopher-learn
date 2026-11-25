package persist

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"path/filepath"
	"strings"
	"testing"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/model"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/online"
)

func testModel(t testing.TB) *model.Model {
	t.Helper()
	rng := rand.New(rand.NewPCG(1, 1))
	var records []model.Record
	for i := range 100 {
		x := rng.Float64()
		color := []string{"red", "blue"}[i%2]
		label := "a"
		if x > 0.5 {
			label = "b"
		}
		records = append(records, model.Record{"x": fmt.Sprint(x), "color": color, "y": label})
	}
	m, _, err := model.Train(context.Background(), records, model.Config{Target: "y", Fit: nn.FitConfig{Epochs: 5}, Rand: rng})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestModelFile(t *testing.T) {
	m := testModel(t)
	path := filepath.Join(t.TempDir(), "model.json")
	if err := SaveModel(path, m); err != nil {
		t.Fatal(err)
	}
	got, err := LoadModel(path)
	if err != nil {
		t.Fatal(err)
	}
	r := model.Record{"x": "0.7", "color": "red"}
	a, _ := m.Predict(r)
	b, err := got.Predict(r)
	if err != nil || a.Label != b.Label || a.Scores["b"] != b.Scores["b"] {
		t.Fatalf("predictions differ: %+v vs %+v (%v)", a, b, err)
	}
	if _, err := LoadModel(filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Error("expected error for missing file")
	}

	// an inconsistent model is rejected on read
	m.Columns = m.Columns[:1]
	var buf bytes.Buffer
	if err := WriteModel(&buf, m); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadModel(&buf); !errors.Is(err, nn.ErrDimension) {
		t.Errorf("inconsistent model: want ErrDimension, got %v", err)
	}
	if _, err := ReadModel(strings.NewReader(`{"kind":"model","version":2,"data":[]}`)); err == nil {
		t.Error("bad payload: expected error")
	}
}

func TestScalerFile(t *testing.T) {
	sc, err := learn.FitScalerVectors([][]float64{{1, 10}, {3, 30}}, learn.MinMax)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "scaler.json")
	if err := SaveScaler(path, sc); err != nil {
		t.Fatal(err)
	}
	got, err := LoadScaler(path)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := got.Transform([]float64{2, 20}); v[0] != 0.5 || v[1] != 0.5 {
		t.Fatalf("scaler not restored: %v", v)
	}
	if _, err := ReadScaler(strings.NewReader(`{"kind":"scaler","version":2,"data":{"offset":[0],"factor":[0]}}`)); err == nil {
		t.Error("invalid scaler: expected error")
	}
	if _, err := ReadScaler(strings.NewReader(`{"kind":"scaler","version":2,"data":7}`)); err == nil {
		t.Error("bad payload: expected error")
	}
}

func TestFormatVersions(t *testing.T) {
	// a network written by gopher-learn v0.4 (format version 1)
	v1 := `{"kind":"network","version":1,"data":{"layers":[{"inputs":1,"outputs":1,"weights":[2],"biases":[1],"activation":"linear"}]}}`
	n, err := ReadNetwork(strings.NewReader(v1))
	if err != nil {
		t.Fatalf("version 1 network must stay readable: %v", err)
	}
	if out, _ := n.Predict([]float64{3}); out[0] != 7 {
		t.Fatalf("version 1 network predicts %v", out)
	}
	// new files carry the current version
	var buf bytes.Buffer
	if err := WriteNetwork(&buf, n); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), `"version":2`) {
		t.Fatalf("written with old version: %s", buf.String())
	}

	for name, doc := range map[string]string{
		"model v1":  `{"kind":"model","version":1,"data":{}}`,
		"scaler v1": `{"kind":"scaler","version":1,"data":{}}`,
		"model v3":  `{"kind":"model","version":3,"data":{}}`,
		"network 0": `{"kind":"network","version":0,"data":{}}`,
		"unknown":   `{"kind":"other","version":2,"data":{}}`,
	} {
		var err error
		switch {
		case strings.HasPrefix(name, "model"), name == "unknown":
			_, err = ReadModel(strings.NewReader(doc))
		case strings.HasPrefix(name, "scaler"):
			_, err = ReadScaler(strings.NewReader(doc))
		default:
			_, err = ReadNetwork(strings.NewReader(doc))
		}
		if !errors.Is(err, ErrFormat) {
			t.Errorf("%s: want ErrFormat, got %v", name, err)
		}
	}
}

func TestOnlineStreamState(t *testing.T) {
	set := learn.NewSet(nn.Classification)
	_ = set.AddClass("a")
	_ = set.AddClass("b")
	o, err := online.New(set, 1, []int{3}, online.DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(2, 2))
	for range 300 {
		x := rng.NormFloat64()
		label := "a"
		if x > 0 {
			label = "b"
		}
		if _, err := o.Learn(learn.NewClassificationSample([]float64{x}, label)); err != nil {
			t.Fatal(err)
		}
	}
	var buf bytes.Buffer
	if err := WriteOnline(&buf, o); err != nil {
		t.Fatal(err)
	}
	got, err := ReadOnline(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if got.Learned != 300 || got.DriftADWIN == nil || got.DriftADWIN.Width != o.DriftADWIN.Width ||
		got.Prequential.Accuracy() != o.Prequential.Accuracy() || got.Config.Memory != online.Window {
		t.Fatalf("stream state not restored: %+v", got)
	}
	if _, err := got.Learn(learn.NewClassificationSample([]float64{1}, "b")); err != nil {
		t.Fatal(err)
	}
}
