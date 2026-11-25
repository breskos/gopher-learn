package nn

import (
	"context"
	"fmt"
	"math/rand/v2"
	"testing"
)

// shapes are typical network sizes: a tiny sensor model, the sonar data set
// and an MNIST-sized image classifier.
var shapes = []struct {
	name   string
	inputs int
	hidden []int
	out    int
}{
	{"tiny-4x8x2", 4, []int{8}, 2},
	{"sonar-60x100x2", 60, []int{100}, 2},
	{"mnist-784x128x64x10", 784, []int{128, 64}, 10},
}

func benchNet(b *testing.B, inputs int, hidden []int, out int, act Activation) *Network {
	b.Helper()
	n, err := New(Config{Inputs: inputs, Hidden: hidden, Outputs: out, HiddenActivation: act, OutputActivation: Softmax, Rand: rand.New(rand.NewPCG(1, 1))})
	if err != nil {
		b.Fatal(err)
	}
	return n
}

func benchData(inputs, out, n int) Data {
	rng := rand.New(rand.NewPCG(2, 2))
	d := Data{X: make([][]float64, n), Y: make([][]float64, n)}
	for i := range n {
		x := make([]float64, inputs)
		for j := range x {
			x[j] = rng.NormFloat64()
		}
		y := make([]float64, out)
		y[rng.IntN(out)] = 1
		d.X[i], d.Y[i] = x, y
	}
	return d
}

func BenchmarkPredict(b *testing.B) {
	for _, s := range shapes {
		b.Run(s.name, func(b *testing.B) {
			n := benchNet(b, s.inputs, s.hidden, s.out, Tanh)
			x := benchData(s.inputs, s.out, 1).X[0]
			b.ReportAllocs()
			for b.Loop() {
				_, _ = n.Predict(x)
			}
		})
	}
}

func BenchmarkTrainBatch32(b *testing.B) {
	for _, s := range shapes {
		b.Run(s.name, func(b *testing.B) {
			n := benchNet(b, s.inputs, s.hidden, s.out, Tanh)
			d := benchData(s.inputs, s.out, 32)
			step := Step{Optimizer: DefaultAdam(), LearningRate: 0.001}
			b.ReportAllocs()
			for b.Loop() {
				_, _ = n.TrainBatchWith(d.X, d.Y, step)
			}
		})
	}
}

func BenchmarkFitEpoch(b *testing.B) {
	for _, s := range shapes {
		b.Run(s.name, func(b *testing.B) {
			n := benchNet(b, s.inputs, s.hidden, s.out, ReLU)
			d := benchData(s.inputs, s.out, 1024)
			cfg := FitConfig{Epochs: 1, BatchSize: 32, LearningRate: 0.001, Rand: rand.New(rand.NewPCG(3, 3))}
			b.ReportAllocs()
			for b.Loop() {
				_, _ = Fit(context.Background(), n, d, Data{}, cfg)
			}
		})
	}
}

// BenchmarkFitEpochSequential is FitEpoch on one goroutine, the setting
// for single-core embedded systems.
func BenchmarkFitEpochSequential(b *testing.B) {
	for _, s := range shapes {
		b.Run(s.name, func(b *testing.B) {
			n := benchNet(b, s.inputs, s.hidden, s.out, ReLU)
			d := benchData(s.inputs, s.out, 1024)
			cfg := FitConfig{Epochs: 1, BatchSize: 32, LearningRate: 0.001, Workers: 1, Rand: rand.New(rand.NewPCG(3, 3))}
			b.ReportAllocs()
			for b.Loop() {
				_, _ = Fit(context.Background(), n, d, Data{}, cfg)
			}
		})
	}
}

func BenchmarkActivation(b *testing.B) {
	z := benchData(256, 1, 1).X[0]
	out := make([]float64, len(z))
	for _, a := range []Activation{Sigmoid, Tanh, ReLU, Softmax} {
		b.Run(fmt.Sprint(a), func(b *testing.B) {
			for b.Loop() {
				a.apply(z, out)
			}
		})
	}
}

func BenchmarkPredictor(b *testing.B) {
	for _, s := range shapes {
		b.Run(s.name, func(b *testing.B) {
			p := benchNet(b, s.inputs, s.hidden, s.out, Tanh).NewPredictor()
			x := benchData(s.inputs, s.out, 1).X[0]
			b.ReportAllocs()
			for b.Loop() {
				_, _ = p.Predict(x)
			}
		})
	}
}

func BenchmarkPredictor32(b *testing.B) {
	for _, s := range shapes {
		b.Run(s.name, func(b *testing.B) {
			p := benchNet(b, s.inputs, s.hidden, s.out, Tanh).Float32().NewPredictor()
			x := make([]float32, s.inputs)
			for i, v := range benchData(s.inputs, s.out, 1).X[0] {
				x[i] = float32(v)
			}
			b.ReportAllocs()
			for b.Loop() {
				_, _ = p.Predict(x)
			}
		})
	}
}
