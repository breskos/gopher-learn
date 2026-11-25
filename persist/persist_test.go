package persist

import (
	"bytes"
	"errors"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"testing"

	"github.com/breskos/gopher-learn/encoders"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/online"
)

func TestNetworkFile(t *testing.T) {
	n, err := nn.New(nn.Config{Inputs: 3, Hidden: []int{4}, Labels: []string{"a", "b"}, Rand: rand.New(rand.NewPCG(1, 1))})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "net.json")
	if err := SaveNetwork(path, n); err != nil {
		t.Fatal(err)
	}
	m, err := LoadNetwork(path)
	if err != nil {
		t.Fatal(err)
	}
	x := []float64{0.1, 0.2, 0.3}
	a, _ := n.Predict(x)
	b, _ := m.Predict(x)
	if !slices.Equal(a, b) {
		t.Fatalf("predictions differ: %v vs %v", a, b)
	}
}

func TestWrongKind(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteSet(&buf, learn.NewSet(nn.Regression)); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadNetwork(&buf); !errors.Is(err, ErrFormat) {
		t.Fatalf("want ErrFormat, got %v", err)
	}
}

func TestOnlineRoundTrip(t *testing.T) {
	set := learn.NewSet(nn.Classification)
	_ = set.AddClass("x")
	_ = set.AddClass("y")
	o, err := online.New(set, 2, []int{3}, online.DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := o.Inject(learn.NewClassificationSample([]float64{1, 2}, "y"), false); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := WriteOnline(&buf, o); err != nil {
		t.Fatal(err)
	}
	o2, err := ReadOnline(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if o2.Data.Len() != 1 || !o2.Data.Contains(o.Data.Samples[0]) {
		t.Fatal("data set not restored")
	}
	if err := o2.Inject(learn.NewClassificationSample([]float64{3, 4}, "x"), false); err != nil {
		t.Fatalf("restored learner cannot inject: %v", err)
	}
}

func TestEncoderRoundTrip(t *testing.T) {
	e := encoders.New("enc")
	in := encoders.NewInput(encoders.String)
	in.AddString("red")
	in.AddString("green")
	if err := e.Fit("color", in); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := WriteEncoder(&buf, e); err != nil {
		t.Fatal(err)
	}
	e2, err := ReadEncoder(&buf)
	if err != nil {
		t.Fatal(err)
	}
	v, err := e2.EncodeString("color", "green")
	if err != nil || !slices.Equal(v, []float64{0, 1}) {
		t.Fatalf("encoded = %v (%v)", v, err)
	}
}
