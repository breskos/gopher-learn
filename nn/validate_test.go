package nn

import (
	"encoding/json"
	"errors"
	"testing"
)

func TestValidate(t *testing.T) {
	valid := func() *Network {
		n, err := New(Config{Inputs: 2, Hidden: []int{3}, Labels: []string{"a", "b"}, Rand: seeded()})
		if err != nil {
			t.Fatal(err)
		}
		return n
	}
	if err := valid().Validate(); err != nil {
		t.Fatalf("valid network rejected: %v", err)
	}
	tests := []struct {
		name   string
		break_ func(n *Network)
	}{
		{"no layers", func(n *Network) { n.Layers = nil }},
		{"nil layer", func(n *Network) { n.Layers[0] = nil }},
		{"bad shape", func(n *Network) { n.Layers[0].Inputs = 0 }},
		{"weights", func(n *Network) { n.Layers[0].Weights = n.Layers[0].Weights[:1] }},
		{"biases", func(n *Network) { n.Layers[1].Biases = nil }},
		{"activation", func(n *Network) { n.Layers[0].Activation = Activation(99) }},
		{"hidden softmax", func(n *Network) { n.Layers[0].Activation = Softmax }},
		{"chain", func(n *Network) {
			n.Layers[1].Inputs = 4
			n.Layers[1].Weights = make([]float64, 4*n.Layers[1].Outputs)
		}},
		{"labels", func(n *Network) { n.Labels = []string{"only one"} }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := valid()
			tt.break_(n)
			if err := n.Validate(); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func TestScores(t *testing.T) {
	n, _ := New(Config{Inputs: 2, Labels: []string{"a", "b"}, OutputActivation: Softmax, Rand: seeded()})
	s, err := n.Scores([]float64{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(s) != 2 || s["a"]+s["b"] < 0.999 || s["a"]+s["b"] > 1.001 {
		t.Fatalf("softmax scores must sum to 1: %v", s)
	}
	if _, err := n.Scores([]float64{1}); !errors.Is(err, ErrDimension) {
		t.Fatalf("want ErrDimension, got %v", err)
	}
	unlabeled, _ := New(Config{Inputs: 2, Outputs: 2})
	if _, err := unlabeled.Scores([]float64{1, 2}); !errors.Is(err, ErrNoLabels) {
		t.Fatalf("want ErrNoLabels, got %v", err)
	}
	if _, err := n.Classify([]float64{1}); !errors.Is(err, ErrDimension) {
		t.Fatalf("Classify: want ErrDimension, got %v", err)
	}
}

func TestTrainErrors(t *testing.T) {
	n, _ := New(Config{Inputs: 2, Outputs: 1, Rand: seeded()})
	if _, err := n.TrainBatch(nil, nil, 0.1); err == nil {
		t.Error("expected error for empty batch")
	}
	if _, err := n.TrainBatch([][]float64{{1, 2}}, nil, 0.1); err == nil {
		t.Error("expected error for mismatched batch")
	}
	if _, err := n.Train([]float64{1}, []float64{1}, 0.1); !errors.Is(err, ErrDimension) {
		t.Errorf("input: want ErrDimension, got %v", err)
	}
	if _, err := n.Train([]float64{1, 2}, []float64{1, 2}, 0.1); !errors.Is(err, ErrDimension) {
		t.Errorf("target: want ErrDimension, got %v", err)
	}
}

func TestReLUTrains(t *testing.T) {
	n, _ := New(Config{Inputs: 1, Hidden: []int{4}, Outputs: 1, HiddenActivation: ReLU, OutputActivation: Linear, Rand: seeded()})
	var first, last float64
	for i := range 500 {
		l, err := n.TrainBatch([][]float64{{0}, {1}, {2}}, [][]float64{{1}, {3}, {5}}, 0.05)
		if err != nil {
			t.Fatal(err)
		}
		if i == 0 {
			first = l
		}
		last = l
	}
	if last >= first/10 {
		t.Fatalf("linear fit did not converge: first %.4f, last %.4f", first, last)
	}
}

func TestEnumText(t *testing.T) {
	for _, a := range []Activation{Sigmoid, Tanh, ReLU, Linear, Softmax} {
		b, err := a.MarshalText()
		if err != nil {
			t.Fatal(err)
		}
		var back Activation
		if err := back.UnmarshalText(b); err != nil || back != a {
			t.Fatalf("activation %v round trip: %v, %v", a, back, err)
		}
	}
	for _, task := range []Task{Classification, Regression} {
		b, err := json.Marshal(task)
		if err != nil {
			t.Fatal(err)
		}
		var back Task
		if err := json.Unmarshal(b, &back); err != nil || back != task {
			t.Fatalf("task %v round trip: %v, %v", task, back, err)
		}
		if task.String() == "" {
			t.Fatal("empty task name")
		}
	}

	if Activation(99).String() != "Activation(99)" || Task(-1).String() != "Task(-1)" {
		t.Error("unexpected name for invalid enum")
	}
	if _, err := Activation(99).MarshalText(); err == nil {
		t.Error("expected error for invalid activation")
	}
	if _, err := Task(5).MarshalText(); err == nil {
		t.Error("expected error for invalid task")
	}
	var a Activation
	if err := a.UnmarshalText([]byte("swish")); err == nil {
		t.Error("expected error for unknown activation")
	}
	var task Task
	if err := task.UnmarshalText([]byte("ranking")); err == nil {
		t.Error("expected error for unknown task")
	}
	if ArgMax(nil) != -1 {
		t.Error("ArgMax of empty slice must be -1")
	}
}
