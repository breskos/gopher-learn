// Command dictionary encodes string states with a dictionary encoder and
// trains a classifier on them. It needs no data files.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/breskos/gopher-learn/encoders"
	"github.com/breskos/gopher-learn/engine"
	"github.com/breskos/gopher-learn/learn"
	"github.com/breskos/gopher-learn/nn"
)

const dimension = "onoff"

// Each row is "<input 1>,<input 2>,<expected label>".
var data = []string{"ON,OFF,ON", "OFF,ON,ON", "OFF,OFF,OFF", "ON,ON,ON"}

func main() {
	enc := encoders.New("answer-representation")
	in := encoders.NewInput(encoders.String)
	for _, row := range data {
		in.AddString(input(row))
	}
	if err := enc.Fit(dimension, in); err != nil { // scans automatically
		log.Fatal(err)
	}
	if err := enc.Explain(os.Stdout); err != nil {
		log.Fatal(err)
	}

	set := learn.NewSet(nn.Classification)
	for _, label := range []string{"ON", "OFF"} {
		if err := set.AddClass(label); err != nil {
			log.Fatal(err)
		}
	}
	for range 50 {
		for _, row := range data {
			vector, err := enc.EncodeString(dimension, input(row))
			if err != nil {
				log.Fatal(err)
			}
			if err := set.Add(learn.NewClassificationSample(vector, label(row))); err != nil {
				log.Fatal(err)
			}
		}
	}

	cfg := engine.DefaultConfig()
	cfg.Decay = 0.05
	e, err := engine.New(set, []int{3}, cfg)
	if err != nil {
		log.Fatal(err)
	}
	if err := e.Run(context.Background(), engine.Distance); err != nil {
		log.Fatal(err)
	}
	network, eval := e.Winner()
	if err := eval.WriteConfusionMatrix(os.Stdout); err != nil {
		log.Fatal(err)
	}

	for _, row := range data {
		vector, err := enc.EncodeString(dimension, input(row))
		if err != nil {
			log.Fatal(err)
		}
		got, err := network.Classify(vector)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%-8s expected %-3s got %s\n", input(row), label(row), got)
	}
}

func input(row string) string {
	parts := strings.Split(row, ",")
	return parts[0] + "," + parts[1]
}

func label(row string) string {
	parts := strings.Split(row, ",")
	return parts[2]
}
