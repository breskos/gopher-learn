package train

import (
	"context"
	"errors"
	"testing"

	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

func TestNewRand(t *testing.T) {
	a, b := NewRand(42), NewRand(42)
	for range 10 {
		if a.Uint64() != b.Uint64() {
			t.Fatal("same seed must produce the same sequence")
		}
	}
	if NewRand(1).Uint64() == NewRand(2).Uint64() {
		t.Fatal("different seeds should produce different sequences")
	}
	if NewRand(0) == nil {
		t.Fatal("seed 0 must return a random generator")
	}
}

func xorSet(t *testing.T) *learn.Set {
	t.Helper()
	s := learn.NewSet(nn.Classification)
	_ = s.AddClass("same")
	_ = s.AddClass("diff")
	for _, x := range [][]float64{{0, 0}, {1, 1}} {
		if err := s.Add(learn.NewClassificationSample(x, "same")); err != nil {
			t.Fatal(err)
		}
	}
	for _, x := range [][]float64{{0, 1}, {1, 0}} {
		if err := s.Add(learn.NewClassificationSample(x, "diff")); err != nil {
			t.Fatal(err)
		}
	}
	return s
}

func TestEpochReducesLoss(t *testing.T) {
	for _, batch := range []int{0, 1, 3, 4, 10} {
		set := xorSet(t)
		rng := NewRand(1)
		net, err := nn.New(nn.Config{Inputs: 2, Hidden: []int{8}, Labels: set.Labels, HiddenActivation: nn.Tanh, OutputActivation: nn.Softmax, Rand: rng})
		if err != nil {
			t.Fatal(err)
		}
		first, err := Epoch(context.Background(), net, set, 0.5, batch, rng)
		if err != nil {
			t.Fatal(err)
		}
		var last float64
		for range 2000 {
			if last, err = Epoch(context.Background(), net, set, 0.5, batch, rng); err != nil {
				t.Fatal(err)
			}
		}
		if last >= first/2 {
			t.Errorf("batch %d: loss did not decrease enough: first %.4f, last %.4f", batch, first, last)
		}
	}
}

func TestEpochEmptySet(t *testing.T) {
	net, _ := nn.New(nn.Config{Inputs: 2, Outputs: 1})
	loss, err := Epoch(context.Background(), net, learn.NewSet(nn.Regression), 0.1, 1, NewRand(1))
	if err != nil || loss != 0 {
		t.Fatalf("empty set: loss %v, err %v", loss, err)
	}
}

func TestEpochCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	set := xorSet(t)
	net, _ := nn.New(nn.Config{Inputs: 2, Labels: set.Labels})
	if _, err := Epoch(ctx, net, set, 0.1, 1, NewRand(1)); !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
}

func TestEpochDimensionError(t *testing.T) {
	set := xorSet(t)
	net, _ := nn.New(nn.Config{Inputs: 3, Labels: set.Labels})
	if _, err := Epoch(context.Background(), net, set, 0.1, 1, NewRand(1)); !errors.Is(err, nn.ErrDimension) {
		t.Fatalf("want ErrDimension, got %v", err)
	}
}
