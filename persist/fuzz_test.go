package persist

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/breskos/gopher-learn/encoders"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/model"
	"github.com/breskos/gopher-learn/nn"
	"github.com/breskos/gopher-learn/online"
)

// FuzzRead feeds arbitrary documents to every reader. A document is either
// rejected with an error or yields an object that works without panicking.
func FuzzRead(f *testing.F) {
	MaxDocumentSize = 1 << 20
	n, _ := nn.New(nn.Config{Inputs: 2, Hidden: []int{2}, Labels: []string{"a", "b"}})
	set := learn.NewSet(nn.Classification)
	_ = set.AddClass("a")
	_ = set.AddClass("b")
	_ = set.Add(learn.NewClassificationSample([]float64{1, 2}, "a"))
	o, _ := online.New(set, 2, []int{2}, online.DefaultConfig())
	enc := encoders.New("e")
	in := encoders.NewInput(encoders.String)
	in.AddString("x y")
	_ = enc.Fit("t", in)
	sc, _ := learn.FitScalerVectors([][]float64{{1}, {2}}, learn.MinMax)
	m := testModel(f)
	for _, write := range []func(*bytes.Buffer) error{
		func(b *bytes.Buffer) error { return WriteNetwork(b, n) },
		func(b *bytes.Buffer) error { return WriteSet(b, set) },
		func(b *bytes.Buffer) error { return WriteOnline(b, o) },
		func(b *bytes.Buffer) error { return WriteEncoder(b, enc) },
		func(b *bytes.Buffer) error { return WriteScaler(b, sc) },
		func(b *bytes.Buffer) error { return WriteModel(b, m) },
	} {
		var b bytes.Buffer
		if err := write(&b); err != nil {
			f.Fatal(err)
		}
		f.Add(b.Bytes())
	}
	f.Fuzz(func(t *testing.T, doc []byte) {
		r := func() *bytes.Reader { return bytes.NewReader(doc) }
		if n, err := ReadNetwork(r()); err == nil {
			_, _ = n.Predict(make([]float64, n.InputSize()))
		}
		if s, err := ReadSet(r()); err == nil && s.Len() > 0 {
			d := s.Data()
			if s.InputSize() > 0 && s.OutputSize() > 0 && s.InputSize() < 1000 && s.OutputSize() < 1000 {
				net, err := nn.New(nn.Config{Inputs: s.InputSize(), Outputs: s.OutputSize()})
				if err == nil {
					_, _ = net.Loss(d.X, d.Y)
				}
			}
		}
		if o, err := ReadOnline(r()); err == nil {
			sample := &learn.Sample{Vector: make([]float64, o.Network.InputSize())}
			if o.Data.Task == nn.Classification && len(o.Data.Labels) > 0 {
				sample.Label = o.Data.Labels[0]
			}
			if o.Network.InputSize() < 1000 && o.Config.MaxEpochs < 50 && o.Config.HotShots < 50 {
				_, _ = o.Learn(sample)
				_, _ = o.Iterate(context.Background())
			}
		}
		if e, err := ReadEncoder(r()); err == nil {
			for name, d := range e.Dimensions {
				v := encoders.Value{String: "x"}
				if d.InputType == encoders.Floats && d.Model != nil && d.Model.Dimensions() < 1000 {
					v = encoders.Value{Floats: make([]float64, 3)}
				}
				_, _ = e.Encode(name, v)
			}
		}
		if s, err := ReadScaler(r()); err == nil {
			_, _ = s.Transform(make([]float64, s.Size()))
		}
		if m, err := ReadModel(r()); err == nil {
			rec := model.Record{}
			for _, c := range m.Columns {
				rec[c.Name] = "1"
			}
			_, _ = m.Predict(rec)
		}
	})
}

func TestMaxDocumentSize(t *testing.T) {
	old := MaxDocumentSize
	defer func() { MaxDocumentSize = old }()
	MaxDocumentSize = 10
	_, err := ReadNetwork(strings.NewReader(`{"kind":"network","version":2,"data":{}}`))
	if err == nil || !strings.Contains(err.Error(), ErrTooLarge.Error()) {
		t.Fatalf("want ErrTooLarge, got %v", err)
	}
}
