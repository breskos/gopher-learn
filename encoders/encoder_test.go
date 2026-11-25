package encoders

import (
	"encoding/json"
	"errors"
	"slices"
	"testing"
)

func fit(t *testing.T, name string, in *Input, typ Type) *Encoder {
	t.Helper()
	e := New("test")
	if err := e.Scan(name, in, typ); err != nil {
		t.Fatal(err)
	}
	if err := e.Fit(name, in); err != nil {
		t.Fatal(err)
	}
	return e
}

func TestFloatReducer(t *testing.T) {
	in := NewInput(Floats)
	in.AddFloats([]float64{1, 0, 3.7, 5})
	in.AddFloats([]float64{2, 0.9, 0.3, 5})
	in.AddFloats([]float64{3, 1.6, 1.3, 5})
	in.AddFloats([]float64{4, 2.9, 4.2, 5})
	e := fit(t, "x", in, FloatReducer)
	// dim 1 is perfectly rank-correlated with dim 0, dim 3 is constant
	for range 20 { // output order must be stable
		v, err := e.EncodeFloats("x", []float64{1, 0, 3.7, 5})
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(v, []float64{1, 3.7}) {
			t.Fatalf("encoded = %v, want [1 3.7]", v)
		}
	}
	if _, err := e.EncodeFloats("x", []float64{1}); err == nil {
		t.Fatal("expected dimension error")
	}
}

func TestDictionary(t *testing.T) {
	in := NewInput(String)
	for _, s := range []string{"ON", "OFF", "on", "Standby"} {
		in.AddString(s)
	}
	e := New("test")
	if err := e.Fit("state", in); err != nil { // automatic scan
		t.Fatal(err)
	}
	if got := e.Dimensions["state"].Type; got != StringDictionary {
		t.Fatalf("scan chose %v", got)
	}
	v, _ := e.EncodeString("state", "on!")
	if !slices.Equal(v, []float64{1, 0, 0}) {
		t.Fatalf("encoded = %v", v)
	}
	v, _ = e.EncodeString("state", "unknown")
	if !slices.Equal(v, []float64{0, 0, 0}) {
		t.Fatalf("encoded unknown = %v", v)
	}
}

func TestSplitDictionary(t *testing.T) {
	in := NewInput(String)
	in.AddString("Wie groß ist das Haus?")
	in.AddString("Was ist ein Garten")
	e := fit(t, "q", in, StringSplitDictionary)
	v, _ := e.EncodeString("q", "Garten Haus")
	tokens := e.Dimensions["q"].Model.(*SplitDictionaryModel).Tokens
	if len(tokens) != 8 || !slices.Contains(tokens, "groß") {
		t.Fatalf("tokens = %v", tokens)
	}
	if sum(v) != 2 {
		t.Fatalf("encoded = %v", v)
	}
}

func TestNGrams(t *testing.T) {
	in := NewInput(String)
	in.AddString("hello")
	in.AddString("help")
	e := fit(t, "w", in, StringNGrams)
	m := e.Dimensions["w"].Model.(*NGramModel)
	if m.Grams[0] != "hel" { // most frequent first
		t.Fatalf("grams = %v", m.Grams)
	}
	v, _ := e.EncodeString("w", "hel")
	if sum(v) != 1 {
		t.Fatalf("encoded = %v", v)
	}
}

func TestJSONRoundTrip(t *testing.T) {
	in := NewInput(String)
	in.AddString("a b")
	in.AddString("c")
	e := fit(t, "s", in, StringSplitDictionary)
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	var e2 Encoder
	if err := json.Unmarshal(b, &e2); err != nil {
		t.Fatal(err)
	}
	a, _ := e.EncodeString("s", "c a")
	c, err := e2.EncodeString("s", "c a")
	if err != nil || !slices.Equal(a, c) {
		t.Fatalf("round trip: %v vs %v (%v)", a, c, err)
	}
}

func TestErrors(t *testing.T) {
	e := New("test")
	if _, err := e.EncodeString("missing", "x"); !errors.Is(err, ErrUnknownDimension) {
		t.Fatalf("want ErrUnknownDimension, got %v", err)
	}
	if err := e.Scan("x", NewInput(String), Automatic); err == nil {
		t.Fatal("expected error for empty input")
	}
	in := NewInput(Floats)
	in.AddFloats([]float64{1})
	if err := e.Scan("x", in, StringNGrams); err == nil {
		t.Fatal("expected type mismatch error")
	}
}

func sum(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x
	}
	return s
}
