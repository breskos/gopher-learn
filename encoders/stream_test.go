package encoders

import (
	"encoding/json"
	"math"
	"slices"
	"testing"
)

func TestHashing(t *testing.T) {
	in := NewInput(String)
	in.AddString("anything")
	m := &HashingModel{}
	cfg := DefaultConfig()
	cfg.HashDimensions = 16
	if err := m.Fit(in, cfg); err != nil {
		t.Fatal(err)
	}
	a, _ := m.Encode(Value{String: "Brand new words!"})
	b, _ := m.Encode(Value{String: "words new brand"})
	if !slices.Equal(a, b) || len(a) != 16 {
		t.Fatalf("hashing depends on order or case: %v vs %v", a, b)
	}
	var sum float64
	for _, x := range a {
		sum += math.Abs(x)
	}
	if sum == 0 || sum > 3 {
		t.Fatalf("three tokens give %v", a)
	}
	if e, _ := m.Encode(Value{String: "  "}); slices.ContainsFunc(e, func(x float64) bool { return x != 0 }) {
		t.Fatal("empty text must encode to zeros")
	}
	cfg.HashDimensions = 0
	if err := m.Fit(in, cfg); err == nil {
		t.Error("zero dimensions: expected error")
	}
	if err := m.Fit(NewInput(Floats), cfg); err == nil {
		t.Error("float input: expected error")
	}
	if e, _ := (&HashingModel{}).Encode(Value{String: "x"}); len(e) != 0 {
		t.Error("unfitted model")
	}
}

func TestTFIDF(t *testing.T) {
	in := NewInput(String)
	for _, s := range []string{"the cat sat", "the dog sat", "the cat ran", "the bird"} {
		in.AddString(s)
	}
	cfg := DefaultConfig()
	m := &TFIDFModel{}
	if err := m.Fit(in, cfg); err != nil {
		t.Fatal(err)
	}
	if want := []string{"bird", "cat", "dog", "ran", "sat", "the"}; !slices.Equal(m.Terms, want) {
		t.Fatalf("terms %v", m.Terms)
	}
	v, _ := m.Encode(Value{String: "the cat"})
	the, cat := v[slices.Index(m.Terms, "the")], v[slices.Index(m.Terms, "cat")]
	if the >= cat {
		t.Fatalf("frequent word weighs %v, rarer word %v", the, cat)
	}
	var norm float64
	for _, x := range v {
		norm += x * x
	}
	if math.Abs(norm-1) > 1e-12 {
		t.Fatalf("not unit length: %v", norm)
	}
	if v, _ := m.Encode(Value{String: "unknown"}); slices.ContainsFunc(v, func(x float64) bool { return x != 0 }) {
		t.Fatal("unknown words must encode to zeros")
	}

	// the vocabulary keeps the terms occurring in most texts
	cfg.TFIDFMaxTerms = 2
	if err := m.Fit(in, cfg); err != nil {
		t.Fatal(err)
	}
	if want := []string{"cat", "the"}; !slices.Equal(m.Terms, want) {
		t.Fatalf("capped terms %v", m.Terms)
	}
	cfg.TFIDFMaxTerms = 0
	if err := m.Fit(in, cfg); err == nil {
		t.Error("zero terms: expected error")
	}
	if err := m.Fit(nil, DefaultConfig()); err == nil {
		t.Error("nil input: expected error")
	}
	if err := json.Unmarshal([]byte(`{"terms":["a"],"idf":[]}`), m); err == nil {
		t.Error("inconsistent model: expected error")
	}
	if err := json.Unmarshal([]byte(`[]`), m); err == nil {
		t.Error("bad JSON: expected error")
	}
}

func TestFloatCategories(t *testing.T) {
	in := NewInput(Floats)
	for _, v := range [][]float64{{3, 1}, {1, 1}, {2, 0}, {3, 0}} {
		in.AddFloats(v)
	}
	m := &FloatCategoriesModel{}
	if err := m.Fit(in, DefaultConfig()); err != nil {
		t.Fatal(err)
	}
	if m.Dimensions() != 5 {
		t.Fatalf("dimensions %d", m.Dimensions())
	}
	if v, _ := m.Encode(Value{Floats: []float64{2, 1}}); !slices.Equal(v, []float64{0, 1, 0, 0, 1}) {
		t.Fatalf("encoded %v", v)
	}
	if v, _ := m.Encode(Value{Floats: []float64{7, 0}}); !slices.Equal(v, []float64{0, 0, 0, 1, 0}) {
		t.Fatalf("unknown code: %v", v)
	}
	if _, err := m.Encode(Value{Floats: []float64{1}}); err == nil {
		t.Error("wrong length: expected error")
	}
	cfg := DefaultConfig()
	cfg.CategoriesMax = 2
	if err := m.Fit(in, cfg); err == nil {
		t.Error("too many categories: expected error")
	}
	bad := NewInput(Floats)
	bad.AddFloats([]float64{math.NaN()})
	if err := m.Fit(bad, DefaultConfig()); err == nil {
		t.Error("NaN: expected error")
	}
	if err := m.Fit(NewInput(String), DefaultConfig()); err == nil {
		t.Error("string input: expected error")
	}
}
