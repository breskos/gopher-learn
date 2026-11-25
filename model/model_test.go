package model

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/breskos/gopher-learn/encoders"
	"github.com/breskos/gopher-learn/explain"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

var colors = []string{"red", "green", "blue"}

// shop returns records where the class depends on a numeric column, a
// large unscaled numeric column and a categorical column.
func shop(rng *rand.Rand, n int) []Record {
	out := make([]Record, n)
	for i := range out {
		x := rng.NormFloat64()
		income := 30000 + rng.Float64()*40000
		color := colors[rng.IntN(len(colors))]
		score := x + (income-50000)/10000
		if color == "red" {
			score += 2
		}
		label := "no"
		if score > 0.5 {
			label = "yes"
		}
		out[i] = Record{"id": strconv.Itoa(i), "x": fmt.Sprint(x), "income": fmt.Sprint(income), "color": color, "buys": label}
	}
	return out
}

// houses returns regression records with a large target.
func houses(rng *rand.Rand, n int) []Record {
	out := make([]Record, n)
	for i := range out {
		rooms := 1 + rng.IntN(6)
		area := 20 + rng.Float64()*180
		city := []string{"berlin", "hamburg", "leipzig"}[rng.IntN(3)]
		price := 50000 + 2500*area + 10000*float64(rooms)
		switch city {
		case "berlin":
			price += 80000
		case "hamburg":
			price += 40000
		}
		price += rng.NormFloat64() * 5000
		out[i] = Record{"rooms": strconv.Itoa(rooms), "area": fmt.Sprint(area), "city": city, "price": fmt.Sprint(price)}
	}
	return out
}

func TestTrainClassification(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	m, h, err := Train(context.Background(), shop(rng, 1200), Config{
		Target: "buys", Ignore: []string{"id"}, Hidden: []int{16}, HiddenActivation: nn.Tanh,
		Fit: nn.FitConfig{Epochs: 150}, Rand: rng,
	})
	if err != nil {
		t.Fatal(err)
	}
	if h.Stop != nn.StopPatience && h.Stop != nn.StopEpochs {
		t.Fatalf("unexpected stop %q", h.Stop)
	}
	want := []Column{{"color", Categorical}, {"income", Numeric}, {"x", Numeric}}
	if fmt.Sprint(m.Columns) != fmt.Sprint(want) {
		t.Fatalf("columns %v, want %v", m.Columns, want)
	}
	if m.Scaler == nil || m.Scaler.Size() != 2 || m.TargetScaler != nil {
		t.Fatal("scaler for two numeric columns expected")
	}
	if got := m.Labels(); fmt.Sprint(got) != "[no yes]" {
		t.Fatalf("labels %v", got)
	}

	e, err := m.Evaluate(shop(rng, 500), 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("accuracy %.3f, AUC %.3f after %d epochs", e.CorrectRatio(), e.MacroAUC(), len(h.Epochs))
	if e.CorrectRatio() < 0.93 || e.MacroAUC() < 0.97 {
		t.Fatalf("accuracy %.3f, AUC %.3f", e.CorrectRatio(), e.MacroAUC())
	}

	p, err := m.Predict(Record{"x": "3", "income": "69000", "color": "red"})
	if err != nil || p.Label != "yes" || p.Scores["yes"] < 0.9 {
		t.Fatalf("clear yes predicted as %+v, %v", p, err)
	}
	// unknown categories and extra columns are fine
	if _, err := m.Predict(Record{"x": "0", "income": "50000", "color": "purple", "other": "?"}); err != nil {
		t.Fatal(err)
	}
}

func TestTrainRegression(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	m, _, err := Train(context.Background(), houses(rng, 1500), Config{
		Target: "price", Task: nn.Regression, Hidden: []int{16}, HiddenActivation: nn.Tanh,
		Fit: nn.FitConfig{Epochs: 200}, Rand: rng,
	})
	if err != nil {
		t.Fatal(err)
	}
	if m.TargetScaler == nil {
		t.Fatal("regression target should be scaled")
	}
	e, err := m.Evaluate(houses(rng, 500), 20000)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("R² %.4f, MAE %.0f, within 20k %.3f", e.R2(), e.MAE(), e.CorrectRatio())
	if e.R2() < 0.97 || e.MAE() > 15000 {
		t.Fatalf("R² %.4f, MAE %.0f", e.R2(), e.MAE())
	}
	p, err := m.Predict(Record{"rooms": "3", "area": "100", "city": "berlin"})
	if err != nil {
		t.Fatal(err)
	}
	if want := 50000 + 250000 + 30000 + 80000.0; math.Abs(p.Value-want) > 30000 {
		t.Fatalf("predicted %.0f, want about %.0f", p.Value, want)
	}
	if p.Label != "" || p.Scores != nil {
		t.Fatal("regression prediction must not have a label")
	}
}

func TestTrainOptions(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	records := shop(rng, 100)
	m, h, err := Train(context.Background(), records, Config{
		Target: "buys", Ignore: []string{"id"}, NoScaling: true, Hidden: []int{},
		Kinds:           map[string]Kind{"x": Categorical},
		Encodings:       map[string]encoders.Type{"color": encoders.StringDictionary},
		Encoder:         &encoders.Config{NGramSize: 2, NGramMaxCapacity: 50, NGramCropRatio: 0.01, DefaultStringEncoder: encoders.StringNGrams},
		ValidationRatio: -1, Fit: nn.FitConfig{Epochs: 3},
	})
	if err != nil {
		t.Fatal(err)
	}
	if m.Scaler != nil || len(m.Network.Layers) != 1 || len(h.Epochs) != 3 || h.Epochs[0].HasVal {
		t.Fatalf("options not applied: scaler %v, %d layers, %d epochs", m.Scaler, len(m.Network.Layers), len(h.Epochs))
	}
	if m.Columns[2] != (Column{"x", Categorical}) || m.Encoder.Dimensions["color"].Type != encoders.StringDictionary {
		t.Fatalf("kinds or encodings not applied: %v", m.Columns)
	}

	m, _, err = Train(context.Background(), records, Config{
		Target: "buys", Ignore: []string{"id", "color"}, Scale: learn.MinMax, Dropout: 0.1, LayerNorm: true, Init: nn.HeNormal,
		Fit: nn.FitConfig{Epochs: 2, Patience: -1}, Rand: rng,
	})
	if err != nil {
		t.Fatal(err)
	}
	if m.Encoder != nil || m.Scaler.Method != learn.MinMax || !m.Network.Layers[0].Norm || m.Network.Layers[0].Dropout != 0.1 {
		t.Fatal("numeric-only model should have no encoder and a min-max scaler")
	}
	if err := m.Validate(); err != nil {
		t.Fatal(err)
	}

	// cancellation is passed through
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := Train(ctx, records, Config{Target: "buys", Rand: rng}); !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
}

func TestTrainErrors(t *testing.T) {
	ok := []Record{{"a": "1", "b": "x", "t": "p"}, {"a": "2", "b": "y", "t": "q"}, {"a": "3", "b": "y", "t": "q"}}
	cases := map[string]struct {
		records []Record
		cfg     Config
	}{
		"no records":         {nil, Config{Target: "t"}},
		"no target":          {ok, Config{}},
		"bad task":           {ok, Config{Target: "t", Task: nn.Task(5)}},
		"unknown target":     {ok, Config{Target: "z"}},
		"unknown ignore":     {ok, Config{Target: "t", Ignore: []string{"z"}}},
		"unknown kind":       {ok, Config{Target: "t", Kinds: map[string]Kind{"z": Numeric}}},
		"invalid kind":       {ok, Config{Target: "t", Kinds: map[string]Kind{"a": Kind(9)}}},
		"numeric encoding":   {ok, Config{Target: "t", Encodings: map[string]encoders.Type{"a": encoders.StringNGrams}}},
		"float encoding":     {ok, Config{Target: "t", Encodings: map[string]encoders.Type{"b": encoders.FloatExact}}},
		"no inputs":          {ok, Config{Target: "t", Ignore: []string{"a", "b"}}},
		"one class":          {[]Record{{"a": "1", "t": "p"}, {"a": "2", "t": "p"}}, Config{Target: "t"}},
		"missing target":     {[]Record{{"a": "1", "t": "p"}, {"a": "2"}}, Config{Target: "t"}},
		"empty target":       {[]Record{{"a": "1", "t": "p"}, {"a": "2", "t": " "}}, Config{Target: "t"}},
		"text target":        {ok, Config{Target: "t", Task: nn.Regression}},
		"text as numeric":    {ok, Config{Target: "t", Kinds: map[string]Kind{"b": Numeric}}},
		"missing numeric":    {[]Record{{"a": "1", "t": "p"}, {"a": "", "t": "q"}}, Config{Target: "t", ValidationRatio: -1}},
		"missing in val":     {[]Record{{"a": "1", "t": "p"}, {"a": "2", "t": "p"}, {"a": "", "t": "q"}, {"a": "4", "t": "q"}, {"a": "", "t": "p"}}, Config{Target: "t", Kinds: map[string]Kind{"a": Numeric}, NoScaling: true, ValidationRatio: 0.5}},
		"bad scale method":   {ok, Config{Target: "t", Scale: learn.ScaleMethod(9)}},
		"bad hidden":         {ok, Config{Target: "t", Hidden: []int{-1}}},
		"bad hidden act":     {ok, Config{Target: "t", HiddenActivation: nn.Softmax}},
		"patience no val":    {ok, Config{Target: "t", ValidationRatio: -1, Fit: nn.FitConfig{Patience: 2}}},
		"no train left":      {[]Record{{"a": "1", "t": "p"}}, Config{Target: "t", Kinds: map[string]Kind{}}},
		"no features at all": {[]Record{{"a": "", "t": "p"}, {"a": " ", "t": "q"}}, Config{Target: "t", Encodings: map[string]encoders.Type{"a": encoders.StringSplitDictionary}, ValidationRatio: -1}},
	}
	for name, c := range cases {
		c.cfg.Rand = rand.New(rand.NewPCG(1, 1))
		c.cfg.Fit.Epochs = 1
		if _, _, err := Train(context.Background(), c.records, c.cfg); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
}

func trained(t *testing.T) *Model {
	t.Helper()
	rng := rand.New(rand.NewPCG(7, 8))
	m, _, err := Train(context.Background(), shop(rng, 200), Config{Target: "buys", Ignore: []string{"id"}, Fit: nn.FitConfig{Epochs: 5}, Rand: rng})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestPredictErrors(t *testing.T) {
	m := trained(t)
	if _, err := m.Predict(Record{"income": "1", "color": "red"}); !errors.Is(err, ErrMissing) || !strings.Contains(err.Error(), `"x"`) {
		t.Errorf("missing column: %v", err)
	}
	if _, err := m.Predict(Record{"x": "  ", "income": "1"}); !errors.Is(err, ErrMissing) {
		t.Errorf("blank value: %v", err)
	}
	for _, bad := range []string{"abc", "NaN", "Inf"} {
		if _, err := m.Predict(Record{"x": bad, "income": "1"}); !errors.Is(err, ErrInvalid) {
			t.Errorf("%q: want ErrInvalid, got %v", bad, err)
		}
	}
	if _, err := m.Evaluate([]Record{{"x": "1", "income": "1", "color": "red"}}, 0); !errors.Is(err, ErrMissing) {
		t.Errorf("Evaluate without target: %v", err)
	}
	if _, err := m.Evaluate([]Record{{"x": "1", "income": "1", "buys": "maybe"}}, 0); !errors.Is(err, ErrInvalid) {
		t.Errorf("Evaluate with unknown class: %v", err)
	}
	if _, err := m.Evaluate([]Record{{"income": "1", "buys": "yes"}}, 0); !errors.Is(err, ErrMissing) {
		t.Errorf("Evaluate with missing input: %v", err)
	}
	m.Task = nn.Regression
	m.Network.Layers[len(m.Network.Layers)-1].Outputs = 1
	if _, err := m.Evaluate([]Record{{"x": "1", "income": "1", "buys": "yes"}}, 0); err == nil {
		t.Error("Evaluate: expected error")
	}
}

func TestValidate(t *testing.T) {
	cases := map[string]func(m *Model){
		"no network":         func(m *Model) { m.Network = nil },
		"invalid network":    func(m *Model) { m.Network.Layers = nil },
		"no columns":         func(m *Model) { m.Columns = nil },
		"invalid kind":       func(m *Model) { m.Columns[0].Kind = Kind(7) },
		"duplicate column":   func(m *Model) { m.Columns[1].Name = m.Columns[2].Name },
		"target as input":    func(m *Model) { m.Columns[1].Name = m.Target },
		"no encoder":         func(m *Model) { m.Encoder = nil },
		"unfitted column":    func(m *Model) { m.Encoder.Dimensions["color"].Model = nil },
		"wrong width":        func(m *Model) { m.Columns = m.Columns[:2] },
		"invalid scaler":     func(m *Model) { m.Scaler.Factor[0] = 0 },
		"scaler size":        func(m *Model) { m.Scaler.Offset, m.Scaler.Factor = []float64{0}, []float64{1} },
		"labels":             func(m *Model) { m.Network.Labels = nil },
		"target scaler":      func(m *Model) { m.TargetScaler = &learn.Scaler{Offset: []float64{0}, Factor: []float64{1}} },
		"regression outputs": func(m *Model) { m.Task = nn.Regression },
		"invalid task":       func(m *Model) { m.Task = nn.Task(4) },
	}
	for name, mutate := range cases {
		m := trained(t)
		mutate(m)
		if err := m.Validate(); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}

	// a regression model with a broken target scaler
	rng := rand.New(rand.NewPCG(9, 9))
	r, _, err := Train(context.Background(), houses(rng, 50), Config{Target: "price", Task: nn.Regression, Fit: nn.FitConfig{Epochs: 1}, Rand: rng})
	if err != nil {
		t.Fatal(err)
	}
	r.TargetScaler.Factor[0] = math.NaN()
	if err := r.Validate(); err == nil {
		t.Error("invalid target scaler: expected error")
	}
	r.TargetScaler = &learn.Scaler{Offset: []float64{0, 0}, Factor: []float64{1, 1}}
	if err := r.Validate(); err == nil {
		t.Error("target scaler size: expected error")
	}
	if _, err := r.Predict(Record{"rooms": "1", "area": "1", "city": "x"}); err == nil {
		t.Error("Predict with a wrong target scaler: expected error")
	}
	r.TargetScaler = nil
	if err := r.Validate(); err != nil {
		t.Errorf("regression without target scaler is valid: %v", err)
	}
	if _, err := r.Predict(Record{"rooms": "1", "area": "1", "city": "x"}); err != nil {
		t.Error(err)
	}
}

func TestEncodeErrors(t *testing.T) {
	m := trained(t)
	m.Scaler.Offset = m.Scaler.Offset[:1]
	if _, err := m.Encode(Record{"x": "1", "income": "1"}); !errors.Is(err, nn.ErrDimension) {
		t.Errorf("scaler mismatch: %v", err)
	}
	m = trained(t)
	m.Encoder = nil
	if _, err := m.Encode(Record{"x": "1", "income": "1"}); err == nil {
		t.Error("missing encoder: expected error")
	}
	m = trained(t)
	delete(m.Encoder.Dimensions, "color")
	if _, err := m.Encode(Record{"x": "1", "income": "1"}); !errors.Is(err, encoders.ErrUnknownDimension) {
		t.Errorf("unknown dimension: %v", err)
	}
	m = trained(t)
	m.Columns = m.Columns[1:]
	if _, err := m.Predict(Record{"x": "1", "income": "1"}); !errors.Is(err, nn.ErrDimension) {
		t.Errorf("network mismatch: %v", err)
	}
}

func TestJSONRoundTrip(t *testing.T) {
	m := trained(t)
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	var got Model
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if err := got.Validate(); err != nil {
		t.Fatal(err)
	}
	r := Record{"x": "0.3", "income": "52000", "color": "blue"}
	a, _ := m.Predict(r)
	c, _ := got.Predict(r)
	if a.Label != c.Label || a.Scores["yes"] != c.Scores["yes"] {
		t.Fatalf("predictions differ: %+v vs %+v", a, c)
	}
}

func TestKindText(t *testing.T) {
	for _, k := range []Kind{Numeric, Categorical} {
		b, err := k.MarshalText()
		if err != nil {
			t.Fatal(err)
		}
		var got Kind
		if err := got.UnmarshalText(b); err != nil || got != k || k.String() != string(b) {
			t.Fatalf("%v: %v, %v", k, got, err)
		}
	}
	if _, err := Kind(5).MarshalText(); err == nil {
		t.Error("expected error")
	}
	if Kind(5).String() != "Kind(5)" {
		t.Error(Kind(5).String())
	}
	var k Kind
	if err := k.UnmarshalText([]byte("nope")); err == nil {
		t.Error("expected error")
	}
}

func TestReadRecords(t *testing.T) {
	recs, err := ReadRecords(strings.NewReader(" a ;b\n1;x\n2;y\n"), WithComma(';'))
	if err != nil {
		t.Fatal(err)
	}
	if len(recs) != 2 || recs[1]["a"] != "2" || recs[0]["b"] != "x" {
		t.Fatalf("records %v", recs)
	}
	if recs, err := ReadRecords(strings.NewReader("a,b\n")); err != nil || len(recs) != 0 {
		t.Fatalf("header only: %v, %v", recs, err)
	}
	for name, in := range map[string]string{
		"empty":      "",
		"no name":    "a,,c\n1,2,3\n",
		"duplicate":  "a,a\n1,2\n",
		"bad header": "\"a\n",
		"short row":  "a,b\n1\n",
	} {
		if _, err := ReadRecords(strings.NewReader(in)); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
	if _, err := LoadRecords(filepath.Join(t.TempDir(), "missing.csv")); err == nil {
		t.Error("LoadRecords: expected error")
	}
}

func FuzzPredict(f *testing.F) {
	rng := rand.New(rand.NewPCG(1, 1))
	m, _, err := Train(context.Background(), shop(rng, 60), Config{Target: "buys", Ignore: []string{"id"}, Fit: nn.FitConfig{Epochs: 1}, Rand: rng})
	if err != nil {
		f.Fatal(err)
	}
	f.Add("1", "50000", "red")
	f.Add("", "1e400", "\x00")
	f.Fuzz(func(t *testing.T, x, income, color string) {
		p, err := m.Predict(Record{"x": x, "income": income, "color": color})
		if err != nil {
			return
		}
		if p.Label != "yes" && p.Label != "no" {
			t.Fatalf("label %q", p.Label)
		}
	})
}

func FuzzReadRecords(f *testing.F) {
	f.Add("a,b\n1,2\n")
	f.Add("a;b\n\"x\";y\n")
	f.Fuzz(func(t *testing.T, in string) {
		recs, err := ReadRecords(strings.NewReader(in))
		if err != nil {
			return
		}
		for _, r := range recs {
			if len(r) != len(recs[0]) {
				t.Fatalf("records differ in width: %v", recs)
			}
		}
	})
}

func TestImportance(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 11))
	records := shop(rng, 1200)
	for _, r := range records {
		r["noise"] = fmt.Sprint(rng.NormFloat64())
	}
	m, _, err := Train(context.Background(), records, Config{Target: "buys", Ignore: []string{"id"}, Hidden: []int{16}, HiddenActivation: nn.Tanh, Fit: nn.FitConfig{Epochs: 100}, Rand: rng})
	if err != nil {
		t.Fatal(err)
	}
	test := shop(rng, 600)
	for _, r := range test {
		r["noise"] = fmt.Sprint(rng.NormFloat64())
	}
	e, _ := m.Evaluate(test, 0)
	t.Logf("accuracy %.3f", e.CorrectRatio())
	imp, err := m.Importance(test, explain.Config{Repeats: 3, Rand: rng})
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range imp {
		t.Logf("%-7s %.3f ± %.3f", f.Feature, f.Mean, f.Std)
	}
	if imp[len(imp)-1].Feature != "noise" || imp[len(imp)-1].Mean > 0.02 || imp[0].Mean < 0.1 {
		t.Fatalf("importances %+v", imp)
	}
	if test[0]["income"] == "" {
		t.Fatal("records were changed")
	}
	if _, err := m.Importance(test[:1], explain.Config{}); err == nil {
		t.Error("one record: expected error")
	}
	delete(test[3], "buys")
	if _, err := m.Importance(test, explain.Config{}); err == nil {
		t.Error("missing target: expected error")
	}
}

// TestImportanceFindsLeak documents the leak recipe of docs/explain.md: a
// column that is only known after the outcome dominates the importance.
func TestImportanceFindsLeak(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 21))
	add := func(rs []Record) []Record {
		for _, r := range rs {
			v := "0" // a voucher is only issued after a purchase
			if r["buys"] == "yes" && rng.Float64() < 0.9 {
				v = fmt.Sprint(5 + rng.IntN(20))
			}
			r["voucher"] = v
			r["noise"] = fmt.Sprint(rng.NormFloat64())
		}
		return rs
	}
	train, test := add(shop(rng, 1500)), add(shop(rng, 600))
	m, _, err := Train(context.Background(), train, Config{Target: "buys", Ignore: []string{"id", "income"}, Hidden: []int{16}, HiddenActivation: nn.Tanh, Fit: nn.FitConfig{Epochs: 100}, Rand: rng})
	if err != nil {
		t.Fatal(err)
	}
	imp, err := m.Importance(test, explain.Config{Repeats: 5, Rand: rng})
	if err != nil {
		t.Fatal(err)
	}
	if imp[0].Feature != "voucher" || imp[0].Mean < 5*imp[1].Mean {
		t.Fatalf("the leak does not stand out: %+v", imp)
	}
}
