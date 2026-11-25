// Command device runs the fault detector exported by ../train on an
// embedded device. It loads the float32 model and classifies sensor
// readings from standard input, one per line:
//
//	temperature,vibration,current,rpm
//
// It imports only the nn package and needs no JSON parser; the prediction
// itself allocates no memory.
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"os"
	"strconv"

	"github.com/breskos/gopher-learn/nn"
)

// scale must match the training (see ../train).
var scale = [4]float32{100, 10, 20, 3000}

func main() {
	model := flag.String("model", "motor.bin", "float32 model written by train")
	flag.Parse()

	data, err := os.ReadFile(*model)
	if err != nil {
		fail(err)
	}
	var network nn.Network32
	if err := network.UnmarshalBinary(data); err != nil {
		fail(err)
	}
	p := network.NewPredictor()
	x := make([]float32, 4)

	in := bufio.NewScanner(os.Stdin)
	out := bufio.NewWriter(os.Stdout)
	defer func() { _ = out.Flush() }()
	for in.Scan() {
		fields := bytes.Split(in.Bytes(), []byte{','})
		if len(fields) != 4 {
			writeLine(out, "error: want temperature,vibration,current,rpm")
			continue
		}
		valid := true
		for i, f := range fields {
			v, err := strconv.ParseFloat(string(bytes.TrimSpace(f)), 32)
			if err != nil {
				valid = false
				break
			}
			x[i] = float32(v) / scale[i]
		}
		if !valid {
			writeLine(out, "error: not a number")
			continue
		}
		label, err := p.Classify(x)
		if err != nil {
			fail(err)
		}
		writeLine(out, label)
	}
}

func writeLine(w *bufio.Writer, s string) {
	if _, err := w.WriteString(s + "\n"); err != nil {
		fail(err)
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "device:", err)
	os.Exit(1)
}
