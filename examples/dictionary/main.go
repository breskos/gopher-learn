package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/breskos/gopher-learn/encoders"
	"github.com/breskos/gopher-learn/engine"
	"github.com/breskos/gopher-learn/learn"
	neural "github.com/breskos/gopher-learn/net"
)

const (
	dataFile      = "data.phrase"
	delimiter     = ","
	tries         = 1
	epochs        = 100
	trainingSplit = 0.7
	learningRate  = 0.4
	decay         = 0.005
	model         = "onoff"
)

func main() {
	e := encoders.NewEncoder("answer-representation")
	input := encoders.NewInput(model, encoders.String)
	input.AddString("ON,OFF")
	input.AddString("OFF,ON")
	input.AddString("OFF,OFF")
	input.AddString("ON,ON")
	e.Scan(model, input, encoders.Automatic)
	e.Transform(model, input)
	e.Explain()

	// TODO(abresk) here I noticed that Sample, Set in learn are not optimized to be used in this manner
	// They were designed to load samples from file.
	// Get vectors for training
	output := make(map[string]int)
	output["ON"] = 0
	output["OFF"] = 1
	learningSet := learn.NewSet(neural.Classification)
	learningSet.AddClass("ON")  // 0
	learningSet.AddClass("OFF") // 1
	data := []string{"ON,OFF,ON", "OFF,ON,ON", "OFF,OFF,OFF", "ON,ON,ON"}
	for i := 0; i < 50; i++ {
		for _, v := range data {
			splitted := strings.Split(v, delimiter)
			input := fmt.Sprintf("%s,%s", splitted[0], splitted[1])
			vector := e.Encode(model, encoders.Unified{String: input})
			outVec := []float64{0.0, 0.0}
			outVec[output[splitted[1]]] = 1.0
			learningSet.AddSample(learn.NewClassificationSample(vector, outVec, splitted[2]))
		}
	}

	// training the network
	en := engine.NewEngine(neural.Classification, []int{3}, learningSet)
	en.SetVerbose(true)
	en.SetConfig(&engine.Config{
		Tries:         tries,
		Epochs:        epochs,
		TrainingSplit: trainingSplit,
		LearningRate:  learningRate,
		Decay:         decay,
	})
	en.Start(neural.Distance)
	network, evaluation := en.GetWinner()
	evaluation.PrintSummary("ON")
	fmt.Println()
	evaluation.PrintSummary("OFF")

	// testing with own example
	for _, v := range data {
		splitted := strings.Split(v, delimiter)
		input := fmt.Sprintf("%s,%s", splitted[0], splitted[1])
		vector := e.Encode(model, encoders.Unified{String: input})
		w := network.CalculateWinnerLabel(vector)
		fmt.Printf("%v -> %v\n", splitted[2], w)
	}
}

func getData() []string {
	file, err := os.Open("data.phrase")

	if err != nil {
		log.Fatalf("failed to open")

	}
	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanLines)
	var text []string
	for scanner.Scan() {
		text = append(text, scanner.Text())
	}
	file.Close()
	return text
}
