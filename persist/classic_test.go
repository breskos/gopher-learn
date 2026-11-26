package persist

import (
	"bytes"
	"math/rand/v2"
	"path/filepath"
	"strings"
	"testing"

	"github.com/breskos/gopher-learn/cluster"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/linear"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/pca"
)

func TestClassicFiles(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	reg := learn.NewSet(nn.Regression)
	cls := learn.NewSet(nn.Classification)
	_ = cls.AddClass("a")
	_ = cls.AddClass("b")
	var points [][]float64
	for i := range 100 {
		x := []float64{rng.NormFloat64(), rng.NormFloat64()}
		_ = reg.Add(learn.NewRegressionSample(x, 2*x[0]-x[1]))
		_ = cls.Add(learn.NewClassificationSample(x, []string{"a", "b"}[i%2]))
		points = append(points, x)
	}
	r, _ := linear.FitRegression(reg, linear.RegressionConfig{})
	l, _ := linear.FitLogistic(cls, linear.LogisticConfig{})
	k, _, _ := cluster.Fit(points, cluster.Config{K: 3})
	p, _ := pca.Fit(points, pca.Config{})
	dir := t.TempDir()
	x := []float64{0.3, -0.2}

	if err := SaveRegression(filepath.Join(dir, "r.json"), r); err != nil {
		t.Fatal(err)
	}
	r2, err := LoadRegression(filepath.Join(dir, "r.json"))
	a, _ := r.Predict(x)
	b, _ := r2.Predict(x)
	if err != nil || a != b {
		t.Fatalf("regression: %v", err)
	}

	if err := SaveLogistic(filepath.Join(dir, "l.json"), l); err != nil {
		t.Fatal(err)
	}
	l2, err := LoadLogistic(filepath.Join(dir, "l.json"))
	pa, _ := l.Probabilities(x)
	pb, _ := l2.Probabilities(x)
	if err != nil || pa[0] != pb[0] {
		t.Fatalf("logistic: %v", err)
	}

	if err := SaveKMeans(filepath.Join(dir, "k.json"), k); err != nil {
		t.Fatal(err)
	}
	k2, err := LoadKMeans(filepath.Join(dir, "k.json"))
	ca, _ := k.Predict(x)
	cb, _ := k2.Predict(x)
	if err != nil || ca != cb {
		t.Fatalf("kmeans: %v", err)
	}

	if err := SavePCA(filepath.Join(dir, "p.json"), p); err != nil {
		t.Fatal(err)
	}
	p2, err := LoadPCA(filepath.Join(dir, "p.json"))
	za, _ := p.Transform(x)
	zb, _ := p2.Transform(x)
	if err != nil || za[0] != zb[0] {
		t.Fatalf("pca: %v", err)
	}

	// invalid and mismatched documents are rejected
	for name, read := range map[string]func(string) error{
		"regression": func(d string) error { _, err := ReadRegression(strings.NewReader(d)); return err },
		"logistic":   func(d string) error { _, err := ReadLogistic(strings.NewReader(d)); return err },
		"kmeans":     func(d string) error { _, err := ReadKMeans(strings.NewReader(d)); return err },
		"pca":        func(d string) error { _, err := ReadPCA(strings.NewReader(d)); return err },
	} {
		kind := map[string]string{"regression": "linear-regression", "logistic": "logistic-regression", "kmeans": "kmeans", "pca": "pca"}[name]
		if err := read(`{"kind":"` + kind + `","version":2,"data":{}}`); err == nil {
			t.Errorf("%s: empty model accepted", name)
		}
		if err := read(`{"kind":"network","version":2,"data":{}}`); err == nil {
			t.Errorf("%s: wrong kind accepted", name)
		}
		if err := read(`{"kind":"` + kind + `","version":1,"data":{}}`); err == nil {
			t.Errorf("%s: version 1 accepted", name)
		}
	}
	var buf bytes.Buffer
	if err := WriteKMeans(&buf, k); err != nil || !strings.Contains(buf.String(), `"kind":"kmeans"`) {
		t.Fatal("kind")
	}
	if _, err := LoadPCA(filepath.Join(dir, "missing.json")); err == nil {
		t.Error("missing file")
	}
}
