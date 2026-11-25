package encoders

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"
)

func TestAllModelsRoundTrip(t *testing.T) {
	strs := NewInput(String)
	for _, s := range []string{"red car", "blue car", "green bike"} {
		strs.AddString(s)
	}
	floats := NewInput(Floats)
	floats.AddFloats([]float64{1, 5, 3})
	floats.AddFloats([]float64{2, 4, 3})
	floats.AddFloats([]float64{3, 1, 3})

	cases := []struct {
		typ   Type
		in    *Input
		value Value
		dims  int
	}{
		{StringDictionary, strs, Value{String: "Blue Car"}, 3},
		{StringSplitDictionary, strs, Value{String: "red bike"}, 5},
		{StringNGrams, strs, Value{String: "car"}, 0},
		{FloatExact, floats, Value{Floats: []float64{7, 8, 9}}, 3},
		{FloatReducer, floats, Value{Floats: []float64{7, 8, 9}}, 1},
	}
	for _, c := range cases {
		t.Run(c.typ.String(), func(t *testing.T) {
			e := New("all")
			if err := e.Scan("x", c.in, c.typ); err != nil {
				t.Fatal(err)
			}
			if err := e.Fit("x", c.in); err != nil {
				t.Fatal(err)
			}
			m := e.Dimensions["x"].Model
			if c.dims > 0 && m.Dimensions() != c.dims {
				t.Fatalf("dimensions = %d, want %d", m.Dimensions(), c.dims)
			}
			want, err := e.Encode("x", c.value)
			if err != nil {
				t.Fatal(err)
			}
			if len(want) != m.Dimensions() {
				t.Fatalf("encoded length %d, dimensions %d", len(want), m.Dimensions())
			}
			b, err := json.Marshal(e)
			if err != nil {
				t.Fatal(err)
			}
			var back Encoder
			if err := json.Unmarshal(b, &back); err != nil {
				t.Fatal(err)
			}
			got, err := back.Encode("x", c.value)
			if err != nil || !slices.Equal(got, want) {
				t.Fatalf("after round trip: %v (%v), want %v", got, err, want)
			}
		})
	}
}

func TestExplain(t *testing.T) {
	e := New("explain")
	in := NewInput(Floats)
	in.AddFloats([]float64{1, 2})
	_ = e.Fit("b", in)
	_ = e.Scan("a", in, FloatExact) // scanned, not fitted
	var buf bytes.Buffer
	if err := e.Explain(&buf); err != nil {
		t.Fatal(err)
	}
	want := "[a] floats input, float-exact, 0 dimensions\n[b] floats input, float-exact, 2 dimensions\n"
	if buf.String() != want {
		t.Fatalf("Explain =\n%s\nwant\n%s", buf.String(), want)
	}
	if _, err := e.EncodeFloats("a", []float64{1, 2}); err == nil {
		t.Fatal("expected error for unfitted dimension")
	}
}

func TestChooseStringType(t *testing.T) {
	cfg := DefaultConfig()
	input := func(values ...string) *Input {
		in := NewInput(String)
		for _, v := range values {
			in.AddString(v)
		}
		return in
	}
	many := func(n int, f func(i int) string) []string {
		out := make([]string, n)
		for i := range out {
			out[i] = f(i)
		}
		return out
	}
	cases := []struct {
		name string
		in   *Input
		want Type
	}{
		{"dictionary", input("on", "off"), StringDictionary},
		{"split dictionary", input(many(60, func(i int) string { return fmt.Sprintf("w%d w%d", i%10, i%7) })...), StringSplitDictionary},
		{"ngrams", input(many(200, func(i int) string { return fmt.Sprintf("x%d", i) })...), StringNGrams},
		{"default", input(many(200, func(i int) string { return fmt.Sprintf("token %d with a rather long text", i) })...), cfg.DefaultStringEncoder},
	}
	for _, c := range cases {
		if got := chooseStringType(cfg, c.in); got != c.want {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
	}
}

func TestNGramCapacity(t *testing.T) {
	in := NewInput(String)
	for i := range 50 {
		in.AddString(fmt.Sprintf("common%03d", i))
	}
	cfg := DefaultConfig()
	cfg.NGramMaxCapacity = 5
	cfg.NGramCropRatio = 0.5
	m := &NGramModel{}
	if err := m.Fit(in, cfg); err != nil {
		t.Fatal(err)
	}
	if len(m.Grams) > 5 {
		t.Fatalf("capacity exceeded: %d grams", len(m.Grams))
	}
	if !slices.Contains(m.Grams, "com") {
		t.Fatalf("most frequent gram missing: %v", m.Grams)
	}
	cfg.NGramSize = 0
	if err := (&NGramModel{}).Fit(in, cfg); err == nil {
		t.Fatal("expected error for NGramSize 0")
	}
	if got := ngrams("ab", 3); got != nil {
		t.Fatalf("ngrams of short string = %v", got)
	}
}

func TestModelInputErrors(t *testing.T) {
	floats := NewInput(Floats)
	floats.AddFloats([]float64{1})
	strs := NewInput(String)
	strs.AddString("x")
	for _, m := range []Model{&DictionaryModel{}, &SplitDictionaryModel{}, &NGramModel{}} {
		if err := m.Fit(floats, DefaultConfig()); err == nil {
			t.Errorf("%T: expected error for float input", m)
		}
		if err := m.Fit(nil, DefaultConfig()); err == nil {
			t.Errorf("%T: expected error for nil input", m)
		}
	}
	for _, m := range []Model{&FloatExactModel{}, &FloatReducerModel{}} {
		if err := m.Fit(strs, DefaultConfig()); err == nil {
			t.Errorf("%T: expected error for string input", m)
		}
	}
	ragged := NewInput(Floats)
	ragged.AddFloats([]float64{1, 2})
	ragged.AddFloats([]float64{1})
	if err := (&FloatExactModel{}).Fit(ragged, DefaultConfig()); err == nil {
		t.Error("expected error for ragged float input")
	}
	exact := &FloatExactModel{Size: 2}
	if _, err := exact.Encode(Value{Floats: []float64{1}}); err == nil {
		t.Error("expected error for wrong float length")
	}
}

func TestEncoderErrors(t *testing.T) {
	if _, err := NewModel(Automatic); err == nil {
		t.Error("NewModel(Automatic): expected error")
	}
	e := &Encoder{} // zero value without map
	in := NewInput(String)
	in.AddString("x")
	if err := e.Scan("s", in, Automatic); err != nil {
		t.Fatalf("Scan on zero-value encoder: %v", err)
	}
	floats := NewInput(Floats)
	floats.AddFloats([]float64{1})
	if err := e.Fit("s", floats); err == nil {
		t.Error("expected error for input type mismatch")
	}
	if err := New("x").Fit("f", NewInput(Floats)); err == nil {
		t.Error("expected error for empty input")
	}
	bad := &Encoder{Config: DefaultConfig(), Dimensions: map[string]*Dimension{"x": {InputType: Floats, Type: Automatic}}}
	if err := bad.Fit("x", floats); err == nil {
		t.Error("expected error for dimension without concrete type")
	}
}

func TestEnumText(t *testing.T) {
	for _, typ := range []Type{Automatic, StringDictionary, StringSplitDictionary, StringNGrams, FloatExact, FloatReducer} {
		b, err := typ.MarshalText()
		if err != nil {
			t.Fatal(err)
		}
		var back Type
		if err := back.UnmarshalText(b); err != nil || back != typ {
			t.Fatalf("type %v round trip: %v, %v", typ, back, err)
		}
	}
	for _, it := range []InputType{String, Floats} {
		b, err := it.MarshalText()
		if err != nil {
			t.Fatal(err)
		}
		var back InputType
		if err := back.UnmarshalText(b); err != nil || back != it {
			t.Fatalf("input type %v round trip: %v, %v", it, back, err)
		}
	}
	if Type(42).String() != "Type(42)" || InputType(9).String() != "InputType(9)" {
		t.Error("unexpected name for invalid enum")
	}
	if _, err := Type(42).MarshalText(); err == nil {
		t.Error("expected error for invalid type")
	}
	if _, err := InputType(9).MarshalText(); err == nil {
		t.Error("expected error for invalid input type")
	}
	var typ Type
	if err := typ.UnmarshalText([]byte("topics")); err == nil {
		t.Error("expected error for unknown type")
	}
	var it InputType
	if err := it.UnmarshalText([]byte("image")); err == nil {
		t.Error("expected error for unknown input type")
	}
}

func TestDimensionJSON(t *testing.T) {
	var d Dimension
	if err := json.Unmarshal([]byte(`{"input_type":"string","type":"string-ngrams","model":null}`), &d); err != nil || d.Model != nil {
		t.Fatalf("null model: %+v, %v", d, err)
	}
	for _, doc := range []string{
		`{"input_type":"string","type":"automatic","model":{}}`,
		`{"input_type":"string","type":"string-ngrams","model":[1]}`,
		`{"input_type":"pixels"}`,
	} {
		if err := json.Unmarshal([]byte(doc), &d); err == nil {
			t.Errorf("%s: expected error", doc)
		}
	}
	for _, m := range []Model{&DictionaryModel{}, &SplitDictionaryModel{}, &NGramModel{}} {
		if err := json.Unmarshal([]byte(`"not an object"`), m); err == nil {
			t.Errorf("%T: expected error for invalid JSON", m)
		}
	}
	if !strings.Contains(normalize("  Hällo,   WELT!! "), "hällo welt") {
		t.Errorf("normalize = %q", normalize("  Hällo,   WELT!! "))
	}
}
