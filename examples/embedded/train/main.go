// Command train trains a small fault detector for a motor on a computer and
// exports it as a float32 model for an embedded device (see ../device).
//
// The sensor data is simulated: temperature (°C), vibration (mm/s), current
// (A) and speed (rpm). A motor is faulty when it runs hot and vibrates, or
// draws too much current for its speed.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"

	"github.com/breskos/gopher-learn/nn"
)

// Feature scaling is part of the export: the device divides by these
// constants, so it needs no scaler.
var scale = [4]float32{100, 10, 20, 3000}

func sample(rng *rand.Rand) ([]float64, int) {
	temp := 40 + rng.Float64()*60
	vib := rng.Float64() * 10
	rpm := 500 + rng.Float64()*2500
	current := 2 + rpm/300 + rng.NormFloat64()
	if rng.Float64() < 0.1 {
		current += 4 // overload
	}
	faulty := temp > 80 && vib > 5 || current > 4+rpm/300
	x := []float64{temp / float64(scale[0]), vib / float64(scale[1]), current / float64(scale[2]), rpm / float64(scale[3])}
	if faulty {
		return x, 1
	}
	return x, 0
}

func main() {
	out := flag.String("out", "motor.bin", "where to write the float32 model")
	flag.Parse()

	rng := rand.New(rand.NewPCG(1, 2))
	data := func(n int) nn.Data {
		var d nn.Data
		for range n {
			x, c := sample(rng)
			y := []float64{0, 0}
			y[c] = 1
			d.X, d.Y = append(d.X, x), append(d.Y, y)
		}
		return d
	}
	train, val, test := data(20000), data(4000), data(4000)

	network, err := nn.New(nn.Config{
		Inputs: 4, Hidden: []int{16, 8}, Labels: []string{"ok", "fault"},
		HiddenActivation: nn.ReLU, OutputActivation: nn.Softmax, Rand: rng,
	})
	if err != nil {
		log.Fatal(err)
	}
	h, err := nn.Fit(context.Background(), network, train, val, nn.FitConfig{Epochs: 200, Patience: 15, Rand: rng})
	if err != nil {
		log.Fatal(err)
	}

	small := network.Float32()
	p64, p32 := network.NewPredictor(), small.NewPredictor()
	x32 := make([]float32, 4)
	correct64, correct32, agree := 0, 0, 0
	for i, x := range test.X {
		want := nn.ArgMax(test.Y[i])
		out64, _ := p64.Predict(x)
		for j, v := range x {
			x32[j] = float32(v)
		}
		out32, _ := p32.Predict(x32)
		c64, c32 := nn.ArgMax(out64), argmax32(out32)
		if c64 == want {
			correct64++
		}
		if c32 == want {
			correct32++
		}
		if c32 == c64 {
			agree++
		}
	}
	n := float64(len(test.X))
	fmt.Printf("trained %d epochs (%s)\n", len(h.Epochs), h.Stop)
	fmt.Printf("test accuracy: float64 %.3f, float32 %.3f, same prediction for %.1f %% of the samples\n",
		float64(correct64)/n, float64(correct32)/n, 100*float64(agree)/n)

	b, err := small.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(*out, b, 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("wrote %s: %d bytes\n", *out, len(b))
}

func argmax32(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}
